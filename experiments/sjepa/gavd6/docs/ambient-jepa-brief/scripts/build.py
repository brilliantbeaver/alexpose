#!/usr/bin/env python3
"""Build the brief in its copied ICLR template and verify artifact integrity."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "iclr/versions/v08"
TMP = ROOT / "tmp/pdfs"
OUT = ROOT / "output/pdf/jepa-gait-research-brief.pdf"
QA = ROOT / "qa"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def main():
    for directory in (TMP, OUT.parent, QA):
        directory.mkdir(parents=True, exist_ok=True)
    # The source study's numerical graphic and template must stay unchanged.
    assert digest(ROOT / "assets/v08-study-questions.pdf") == digest(SOURCE / "figures/study-questions.pdf")
    for path in (ROOT / "template").iterdir():
        if path.is_file():
            assert digest(path) == digest(SOURCE / "template" / path.name), path.name
    provenance = json.loads((ROOT / "assets/external/PROVENANCE.json").read_text())
    assert digest(ROOT / "assets/external/vjepa-figure2.pdf") == provenance["figure_sha256"]
    print(run("tectonic", "--keep-logs", "--keep-intermediates", "--outdir", str(TMP), "manuscript.tex"))
    log = (TMP / "manuscript.log").read_text()
    for failure in ("Overfull", "Undefined control sequence", "There were undefined references", "There were undefined citations"):
        assert failure not in log, failure
    run("pdftotext", "-layout", str(TMP / "manuscript.pdf"), str(TMP / "manuscript.txt"))
    pages = [page for page in (TMP / "manuscript.txt").read_text().split("\f") if page.strip()]
    assert len(pages) == 4, f"Expected 3 body pages and 1 reference page, found {len(pages)}"
    assert "REFERENCES" in re.sub(r"\s+", "", pages[3])
    assert "Figure 1:" in pages[0] and "Figure 2:" in pages[1]
    assert "Landay" in pages[2] and "Delp" in pages[2]
    assert "Published as a conference paper" not in "".join(pages)
    shutil.copy2(TMP / "manuscript.pdf", OUT)
    run("pdftoppm", "-scale-to", "1600", "-png", str(OUT), str(QA / "final"))
    (QA / "font-report.txt").write_text(run("pdffonts", str(OUT)))
    (QA / "verification.json").write_text(json.dumps({
        "body_pages": 3,
        "reference_pages": 1,
        "figures": {"external_concept_page": 1, "unchanged_v08_results_page": 2},
        "template_byte_identical_to_v08": True,
        "result_asset_byte_identical_to_v08": True,
        "external_asset_matches_provenance": True,
        "no_overfull_boxes_or_unresolved_references": True,
        "source_pdf_sha256": digest(SOURCE / "paper-v08.pdf"),
        "manuscript_sha256": digest(ROOT / "manuscript.tex"),
        "output_pdf_sha256": digest(OUT),
        "visual_review": "See reviews/FINAL-DISPOSITION.md; automated checks do not replace visual inspection."
    }, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
