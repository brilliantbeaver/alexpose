"""Check manuscript packaging and render all versions for author-side inspection.

Set TECTONIC and PDFTOPPM if they are not on PATH. This does not run ML inference.
The matching .tex and private assets, rather than shared tools, freeze each version.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

from PIL import Image, ImageDraw
from pypdf import PdfReader
from verify_summary import verify

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(version):
    assets = ROOT / f"assets/v{version}"
    prefix = ROOT / f"paper_v{version}"
    pdf = prefix.with_suffix(".pdf")
    verify(assets / "numerical_evidence.json")
    reader = PdfReader(pdf)
    texts = [p.extract_text() for p in reader.pages]
    alltext = "\n".join(texts)
    references = next(i for i, t in enumerate(texts) if "References" in t)
    assert references == 4
    assert "??" not in alltext and "Anonymous Author(s)" in alltext
    assert not re.search(r"/Users/|theodoremui|gavd5-drift|physworld_revisions|fmts_revisions", alltext)
    assert reader.metadata.get("/Author", "") in ("", None)
    logfile = prefix.with_suffix(".log")
    if not logfile.exists():
        logfile = assets / "build" / logfile.name
    log = logfile.read_text()
    assert not re.search(r"Overfull|undefined references|Missing character", log)
    package = ROOT / f"paper_v{version}_overleaf.zip"
    tectonic = os.getenv("TECTONIC") or shutil.which("tectonic")
    with tempfile.TemporaryDirectory(prefix=f"fmts_v{version}_") as scratch:
        scratch = Path(scratch)
        with zipfile.ZipFile(package) as z:
            assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
            for name in z.namelist():
                if name.endswith((".tex", ".bib", ".svg", ".txt")):
                    value = z.read(name).decode()
                    assert not re.search(r"/Users/|theodoremui|alexpose|gavd5-drift|physworld_revisions", value)
            z.extractall(scratch)
        proc = subprocess.run([tectonic, "--keep-logs", "main.tex"], cwd=scratch, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout+proc.stderr
        rebuilt = [p.extract_text() for p in PdfReader(scratch / "main.pdf").pages]
        assert rebuilt == texts, "Isolated source package differs from delivered PDF"
        (assets / "isolated_build.txt").write_text(proc.stdout+proc.stderr)
    supplement = ROOT / f"paper_v{version}_supplement.zip"
    # The aggregate verifier is identical across versions and contains no local paths.
    shutil.copy2(ROOT / "tools/verify_summary.py", assets / "verify_summary.py")
    with zipfile.ZipFile(supplement, "w", zipfile.ZIP_DEFLATED) as z:
        for filename in ("numerical_evidence.json", "draw_figures.py", "verify_summary.py"):
            z.write(assets / filename, filename)
        z.write(assets / "figures/FIGURE_NOTES.md", "FIGURE_NOTES.md")
        z.writestr("README.md", "# Anonymous numerical supplement\n\nRetained seed aggregates and recorded source intervals accompany the manuscript. Run `python verify_summary.py` to check means and paired differences. Raw poses, predictions and checkpoints are absent: the source-bootstrap intervals and encoder inference cannot be reproduced from these aggregates.\n\nInstall ReportLab and run `python draw_figures.py --version "+str(version)+" --assets .` to regenerate the SVG/PDF figures. The poses are schematic, not participant examples or forecasts. The source package separately contains the manuscript, bibliography, official style and frozen figures. No new model training was performed.\n")
    render = assets / "rendered"
    render.mkdir(exist_ok=True)
    poppler = os.getenv("PDFTOPPM") or shutil.which("pdftoppm")
    proc = subprocess.run([poppler, "-scale-to", "1000", "-png", str(pdf), str(render / "page")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    paths = sorted(render.glob("page-*.png"))
    assert len(paths) == len(texts)
    # Two-page strips preserve legibility better than a very tall contact sheet.
    for offset in range(0, len(paths), 2):
        images = [Image.open(p).convert("RGB") for p in paths[offset:offset+2]]
        sheet = Image.new("RGB", (sum(i.width for i in images), 1035), "#e9edf0")
        x = 0
        for index, im in enumerate(images):
            sheet.paste(im, (x, 35))
            ImageDraw.Draw(sheet).text((x+20, 10), f"FMTS V{version} / page {offset+index+1}", fill="#172b3a")
            x += im.width
        sheet.save(render / f"spread-{offset//2+1}.jpg", quality=90)
    return {"version": version, "main_pages": references, "total_pages": len(texts),
            "anonymous_source_package": "passed", "isolated_compile_text_matches": True,
            "aggregate_arithmetic": "passed", "undefined_references_or_overfull_boxes": False,
            "rendered_pages": len(paths), "visual_inspection": "pending",
            "pdf_sha256": digest(pdf), "source_zip_sha256": digest(package), "supplement_sha256": digest(supplement)}


if __name__ == "__main__":
    # Independent version directories; no common TeX or figure files are mutated.
    with ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(check, range(1, 9)))
    (ROOT / "review/delivery_checks.json").write_text(json.dumps(records, indent=2)+"\n")
    print(json.dumps(records, indent=2))
