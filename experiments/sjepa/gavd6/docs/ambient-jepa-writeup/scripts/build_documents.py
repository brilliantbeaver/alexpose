"""Build readable HTML views and a vector figure atlas; no external resources."""
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parents[1]
for source,target,title in [("PLAN.md","plan.html","JEPA gait research: writing and research plan"),
                            ("FIGURES.md","figures.html","JEPA gait research: figure gallery")]:
    subprocess.run(["pandoc",source,"--standalone","--toc","--toc-depth=2","--mathml",
                    "--css=design.css","--metadata",f"title={title}","-o",target],cwd=HERE,check=True)
    # Use the readable companion views for navigation while preserving Markdown sources.
    p=HERE/target
    p.write_text(p.read_text().replace('href="PLAN.md"','href="plan.html"').replace('href="FIGURES.md"','href="figures.html"'))
pdfs=sorted((HERE/"figures").glob("0[1-8]-*.pdf"))
assert len(pdfs)==8
out=HERE/"output/pdf";out.mkdir(parents=True,exist_ok=True)
subprocess.run(["pdfunite",*[str(p) for p in pdfs],str(out/"jepa-gait-figure-atlas.pdf")],check=True)
print("Built plan.html, figures.html, and output/pdf/jepa-gait-figure-atlas.pdf")
