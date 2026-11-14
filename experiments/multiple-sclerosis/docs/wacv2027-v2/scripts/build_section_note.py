"""Build the standalone v1 Section 4.1 refinement from its Markdown source."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAME = 'wacv2007-v1-section-4.1-refined'
QA = ROOT / 'qa' / NAME
QA.mkdir(parents=True, exist_ok=True)
subprocess.run(['pandoc', str(ROOT / (NAME + '.md')), '--standalone',
                '--from=markdown', '--to=latex',
                '-o', str(ROOT / (NAME + '.tex'))], check=True)
tex_path = ROOT / (NAME + '.tex')
source = tex_path.read_text()
# Keep the complete replacement on page one, without a large title-page gap.
compact_title = r'''\makeatletter
\renewcommand{\maketitle}{\noindent{\LARGE\@title\par}\vspace{0.6em}
\noindent{\small\@date\par}\vspace{1em}}
\makeatother
'''
source = source.replace(r'\begin{document}', compact_title + '\n' + r'\begin{document}')
for heading in ['What required', 'Evidence and']:
    source = source.replace(r'\subsection{' + heading, r'\clearpage' + '\n' + r'\subsection{' + heading)
tex_path.write_text(source)
result = subprocess.run(['tectonic', '--keep-logs', '--outdir', str(QA),
                         str(ROOT / (NAME + '.tex'))], cwd=ROOT,
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(QA / 'build-output.txt').write_text(result.stdout)
if result.returncode:
    raise RuntimeError(result.stdout)
shutil.copyfile(QA / (NAME + '.pdf'), ROOT / (NAME + '.pdf'))
subprocess.run(['pdftotext', '-layout', str(ROOT / (NAME + '.pdf')),
                str(QA / 'text.txt')], check=True)
for old in QA.glob('page-*.png'):
    old.unlink()
subprocess.run(['pdftoppm', '-scale-to', '1200', '-png',
                str(ROOT / (NAME + '.pdf')), str(QA / 'page')], check=True)
print('Built', NAME + '.pdf')
