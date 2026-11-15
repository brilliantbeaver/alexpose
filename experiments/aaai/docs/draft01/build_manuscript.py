"""Build the anonymous manuscript with the unmodified official AAAI-27 style."""
from pathlib import Path
import os,shutil,subprocess,json,hashlib
BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'ambient').is_dir())
from dotenv import load_dotenv as load_env,find_dotenv as find_env
os.chdir(ROOT);load_env(find_env());os.chdir(BASE)
env=os.environ.copy()
texhome=os.getenv('AAAI_TEXMF_HOME')
if not texhome and Path('/tmp/aaai-texmf').exists():texhome='/tmp/aaai-texmf'
if texhome:
    env.update(TEXMFHOME=texhome,TEXMFVAR=os.getenv('AAAI_TEXMF_VAR','/tmp/aaai-texvar'),TEXMFCONFIG=os.getenv('AAAI_TEXMF_CONFIG','/tmp/aaai-texconfig'))
pdftex=shutil.which('pdflatex') or '/Library/TeX/texbin/pdflatex'
bibtex=shutil.which('bibtex') or str(Path(pdftex).with_name('bibtex'))
pandoc=shutil.which('pandoc')
if not pandoc or not Path(pdftex).exists():raise RuntimeError('Install pandoc and pdfLaTeX; the official style does not support XeTeX')
for name in ('aaai2027.sty','aaai2027.bst'):
    source=BASE/'author_kit'/name;target=BASE/name
    if target.exists() and target.read_bytes()!=source.read_bytes():raise RuntimeError('Official style was modified')
    shutil.copy2(source,target)
commands=[[pandoc,'manuscript.md','--from=markdown-auto_identifiers+raw_tex','--to=latex','--standalone','--natbib','--bibliography=references.bib','--template=template.tex','--wrap=none','--output=manuscript.tex'],[pdftex,'-interaction=nonstopmode','-halt-on-error','manuscript.tex'],[bibtex,'manuscript'],[pdftex,'-interaction=nonstopmode','-halt-on-error','manuscript.tex'],[pdftex,'-interaction=nonstopmode','-halt-on-error','manuscript.tex']]
logs=BASE/'build_logs';logs.mkdir(exist_ok=True)
for i,command in enumerate(commands):
    with (logs/f'{i:02d}_{Path(command[0]).name}.log').open('w') as output:result=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Build failed at step{i}; inspect {logs}')
record={'pdf_sha256':hashlib.sha256((BASE/'manuscript.pdf').read_bytes()).hexdigest(),'official_style_unchanged':all((BASE/n).read_bytes()==(BASE/'author_kit'/n).read_bytes() for n in ('aaai2027.sty','aaai2027.bst')),'compiler':pdftex,'commands':commands}
(BASE/'build_record.json').write_text(json.dumps(record,indent=2));print(BASE/'manuscript.pdf')
