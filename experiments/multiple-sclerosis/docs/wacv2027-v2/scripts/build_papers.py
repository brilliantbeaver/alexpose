"""Build immutable manuscript revisions with the official WACV 2027 style."""
from pathlib import Path
import argparse
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'template/wacv-2027-author-kit-template'

def build(version):
    name = f'wacv2007-v{version}'
    destination = ROOT / 'qa' / name
    destination.mkdir(parents=True, exist_ok=True)
    # Keep each revision's bibliography reproducible as later drafts evolve.
    bibliography = ROOT / f'references-v{version}.bib'
    if not bibliography.exists():
        shutil.copyfile(ROOT / 'references.bib', bibliography)
    manuscript = ROOT / f'{name}.tex'
    text = manuscript.read_text()
    import re
    text = re.sub(r'\\bibliography\{references(?:-v\d+)?\}',
                  lambda _: r'\bibliography{references-v' + str(version) + '}', text)
    manuscript.write_text(text)
    for filename in ('wacv.sty', 'ieeenat_fullname.bst'):
        shutil.copyfile(TEMPLATE / filename, ROOT / filename)
    command = ['tectonic', '--keep-logs', '--keep-intermediates', '--outdir', str(destination), str(ROOT / f'{name}.tex')]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (destination / 'build-output.txt').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout)
    shutil.copyfile(destination / f'{name}.pdf', ROOT / f'{name}.pdf')
    subprocess.run(['pdftotext', '-layout', str(ROOT / f'{name}.pdf'), str(destination / 'text.txt')], check=True)
    for old_render in destination.glob('page-*.png'):
        old_render.unlink()
    subprocess.run(['pdftoppm', '-scale-to', '1200', '-png', str(ROOT / f'{name}.pdf'), str(destination / 'page')], check=True)
    print(f'Built {name}.pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('versions', nargs='+', type=int)
    for version in parser.parse_args().versions:
        build(version)
