"""Inspect and rasterize PDFs using the separately recorded PyMuPDF runtime."""
from pathlib import Path
import hashlib,json
import pymupdf as fitz
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/restart_pdf_check'
def main():
    OUT.mkdir(exist_ok=True);report={}
    for name,path in [('slides',ROOT/'slides/restart.pdf'),('manuscript',ROOT/'docs/draft01/manuscript.pdf')]:
        d=fitz.open(path);issues=[];fonts=set()
        for i,page in enumerate(d):
            fonts.update((font[1],font[2],font[3]) for font in page.get_fonts())
            scale=.7 if name=='slides' else 1
            page.get_pixmap(matrix=fitz.Matrix(scale,scale)).save(OUT/f'{name}-{i+1:02d}.png')
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        x0,y0,x1,y1=span['bbox']
                        if x0<0 or y0<0 or x1>page.rect.width+.5 or y1>page.rect.height+.5:issues.append({'page':i+1,'text':span['text']})
        assert not issues,(name,issues)
        assert all(font[1]!='Type3' for font in fonts),(name,'Type3 fonts')
        report[name]=dict(pages=len(d),off_page_text=issues,type3_fonts=False,fonts=sorted(fonts),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report['runtime']=fitz.VersionBind
    text=json.dumps(report,indent=2)+'\n';(OUT/'render_verification.json').write_text(text)
    (ROOT/'docs/restart/pdf_validation.json').write_text(text)
    print(json.dumps({k:v.get('pages') for k,v in report.items() if isinstance(v,dict)}))
if __name__=='__main__':main()
