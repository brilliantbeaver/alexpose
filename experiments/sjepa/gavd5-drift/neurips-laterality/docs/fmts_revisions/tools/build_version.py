"""Build one self-contained anonymous FMTS version, without editing its Markdown.

Requires pandoc, tectonic, reportlab and pypdf. Set TECTONIC if not on PATH.
Each version has private copies of figures, evidence, bibliography and tools.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,zipfile
from pypdf import PdfReader

HERE=Path(__file__).resolve().parents[1]

def run(args,**kw):
    p=subprocess.run(args,text=True,capture_output=True,**kw)
    if p.returncode: raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout+p.stderr

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('version',type=int);ap.add_argument('--init-assets',action='store_true');args=ap.parse_args()
    v=args.version;assets=HERE/f'assets/v{v}';md=HERE/f'paper_v{v}.md'
    if args.init_assets:
        assets.mkdir(parents=True,exist_ok=False)
        shutil.copy2(HERE/'review/numerical_evidence.json',assets/'numerical_evidence.json')
        shutil.copy2(HERE/'review/references_verified.bib',assets/'references.bib')
        shutil.copy2(HERE/'template/official/neurips_2026.sty',assets/'neurips_2026.sty')
        shutil.copy2(HERE/'tools/draw_figures.py',assets/'draw_figures.py')
        keyfile=HERE/f'review/citation_keys_v{v}.json'
        if keyfile.exists():shutil.copy2(keyfile,assets/'citation_keys.json')
        run([os.sys.executable,str(assets/'draw_figures.py'),'--version',str(v),'--assets',str(assets)])
    source=md.read_text();title,body=source.split('\n',1);title=title.removeprefix('# ')
    abstract,rest=body.strip().removeprefix('## Abstract\n').split('\n## ',1)
    rest='## '+rest
    mainbody,appendix=rest.split('\n## Appendix A.',1)
    if '\n## References' in mainbody:mainbody=mainbody.split('\n## References',1)[0]
    # Turn image captions into ordinary NeurIPS figures, preserving SVG in MD.
    def figure(m):
        caption,path=m.groups();return '\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{'+path.replace('.svg','.pdf')+'}\n\\caption{'+caption+'}\n\\end{figure}\n'
    def convert(text):
        if (assets/'citation_keys.json').exists():
            keys=json.loads((assets/'citation_keys.json').read_text())
            text=re.sub(r'\[(\d+)\]',lambda m: r'\citep{'+keys[m.group(1)]+'}',text)
        text=re.sub(r'!\[([^\n]*)\]\(([^)]+)\)',figure,text)
        def compact_table(match):
            lines=match.group(0).strip().splitlines()
            rows=[[x.strip().replace('%',r'\%') for x in line.strip('|').split('|')] for line in lines]
            rows=[row for row in rows if not all(re.fullmatch(r'[: -]+',x) for x in row)]
            return '\\begin{center}\n\\begin{tabular}{lrrl}\n\\toprule\n'+' \\\\\n'.join(' & '.join(row) for row in rows[:1])+' \\\\\n\\midrule\n'+' \\\\\n'.join(' & '.join(row) for row in rows[1:])+' \\\\\n\\bottomrule\n\\end{tabular}\n\\end{center}\n'
        text=re.sub(r'(?m)^\|[^\n]+\n(?:\|[^\n]+\n?)+',compact_table,text)
        return run(['pandoc','--from','markdown+tex_math_dollars+raw_tex','--to','latex','--natbib','--shift-heading-level-by=-1'],input=text)
    pre=r'''\documentclass{article}
\PassOptionsToPackage{numbers,sort&compress}{natbib}
\usepackage[dblblindworkshop]{neurips_2026}
\workshoptitle{Foundation Models for Temporal Systems: From Forecasting to World Modeling}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,calc}
\usepackage{microtype,url}
\usepackage[hidelinks,unicode]{hyperref}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\setlength{\emergencystretch}{1em}
'''
    tex=pre+'\\title{'+title+'}\n\\author{Anonymous Author(s)}\n\\hypersetup{pdftitle={'+title+'},pdfauthor={},pdfcreator={LaTeX}}\n\\begin{document}\n\\maketitle\n\\begin{abstract}\n'+convert(abstract)+'\\end{abstract}\n'+convert(mainbody)
    tex+='\n\\clearpage\n\\bibliographystyle{unsrtnat}\n\\bibliography{assets/v'+str(v)+'/references}\n\\clearpage\n\\appendix\n'+convert('## '+appendix.replace('\n## Appendix B.','\n## '))+'\n\\end{document}\n'
    (HERE/f'paper_v{v}.tex').write_text(tex)
    # A local style copy is byte-identical to the official download.
    if not (HERE/'neurips_2026.sty').exists():shutil.copy2(assets/'neurips_2026.sty',HERE/'neurips_2026.sty')
    tectonic=os.getenv('TECTONIC') or shutil.which('tectonic')
    if not tectonic:raise RuntimeError('Set TECTONIC to the Tectonic executable')
    log=run([tectonic,'--keep-logs','--keep-intermediates',f'paper_v{v}.tex'],cwd=HERE)
    (assets/'build_stdout.txt').write_text(log)
    reader=PdfReader(HERE/f'paper_v{v}.pdf');texts=[p.extract_text() for p in reader.pages]
    reference_page=next(i+1 for i,t in enumerate(texts) if 'References' in t)
    local_extras=['assets/', '/Users/', 'C:\\Users', 'theodoremui','alexm@','Physical World AI','paper_v']
    assert not any(x in '\n'.join(texts) for x in local_extras)
    assert '??' not in '\n'.join(texts)
    # Compile source package uses a self-contained private asset directory.
    package=HERE/f'paper_v{v}_overleaf.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('main.tex',tex.replace(f'assets/v{v}/','assets/'))
        z.write(assets/'neurips_2026.sty','neurips_2026.sty')
        z.write(assets/'references.bib','assets/references.bib')
        for p in (assets/'figures').glob('*'):
            if p.suffix in {'.svg','.pdf'}:z.write(p,'assets/figures/'+p.name)
        z.writestr('README.txt','Anonymous FMTS manuscript. Compile main.tex with pdfLaTeX + BibTeX, or Tectonic. Four-page workshop main-text limit; references and appendices excluded. No shell escape required.\n')
    with zipfile.ZipFile(HERE/f'paper_v{v}_supplement.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['numerical_evidence.json','draw_figures.py']:
            z.write(assets/name,name)
        z.write(assets/'figures/FIGURE_NOTES.md','FIGURE_NOTES.md')
        z.writestr('README.md','# Anonymous numerical supplement\n\nThis package contains retained aggregate results, not raw poses, predictions or checkpoints. Source-bootstrap intervals cannot be reconstructed from seed-level scores. Install reportlab, then run `python draw_figures.py --version '+str(v)+' --assets .` to regenerate vector figures. The manuscript appendix describes preprocessing, training, splitting and readout selection. Source datasets and trained models are required to rerun the experiment.\n')
    record={'version':v,'main_pages':reference_page-1,'references_start_page':reference_page,'total_pages':len(texts),'official_style_sha256':sha(assets/'neurips_2026.sty'),'markdown_sha256':sha(md),'tex_sha256':sha(HERE/f'paper_v{v}.tex'),'pdf_sha256':sha(HERE/f'paper_v{v}.pdf'),'overleaf_sha256':sha(package),'font_size_or_margin_reduction':False,'visual_inspection':'pending','files':{str(p.relative_to(assets)):sha(p) for p in assets.rglob('*') if p.is_file() and p.name!='manifest.json'}}
    (assets/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['version','main_pages','total_pages']}))

if __name__=='__main__':main()
