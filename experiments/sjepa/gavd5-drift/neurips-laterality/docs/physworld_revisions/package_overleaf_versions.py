"""Validate PDFs and assemble one self-contained Overleaf ZIP per revision.

Run build_submission_versions.py first. ZIPs contain generated TeX, the official
style, only the figure PDFs used by that revision, and a short upload guide.
No research data, model checkpoints, review notes, or personal files are bundled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
WORKSPACE = DOCS.parents[1]
BUILD = WORKSPACE / "tmp" / "pdfs" / "physworld-submissions"
CHECK = WORKSPACE / "tmp" / "pdfs" / "physworld-overleaf-check"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_page(aux: str, label: str):
    match = re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
    return int(match.group(1)) if match else None


def check_pdf(path: Path, aux: str, tex: str) -> dict:
    reader = PdfReader(path)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(page_texts)
    if not reader.metadata.title or reader.metadata.author != "Anonymous Author(s)":
        raise ValueError(f"Incorrect PDF metadata: {path.name}")
    for forbidden in ["C:\\Users", "C:/Users", "alexm", "??", "Source linked in the original revision"]:
        if forbidden in text:
            raise ValueError(f"Unexpected PDF content {forbidden!r}: {path.name}")
    for page in reader.pages:
        if tuple(round(float(x), 2) for x in page.mediabox[2:]) != (612.0, 792.0):
            raise ValueError(f"Non-letter page in {path.name}")
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A", {})
            if action.get("/S") in {"/Launch", "/GoToR", "/JavaScript"}:
                raise ValueError(f"Local or executable PDF action in {path.name}")
            uri = str(action.get("/URI", ""))
            if uri and not uri.startswith(("https://", "http://")):
                raise ValueError(f"Non-web PDF URI {uri!r}")
    references_start = label_page(aux, "physworld:references-start")
    appendix_start = label_page(aux, "physworld:appendix-start")
    main_pages = references_start - 1
    return {"total_pages": len(reader.pages), "main_pages": main_pages,
            "references_start_page": references_start,
            "appendix_start_page": appendix_start,
            "within_eight_main_pages": main_pages <= 8,
            "title_metadata": reader.metadata.title,
            "author_metadata": reader.metadata.author,
            "source_tables": tex.count(r"\begin{table}"),
            "breakable_longtables": tex.count(r"\begin{longtable}"),
            "figure_count": tex.count(r"\includegraphics"),
            "bibliography_entries": tex.count(r"\bibitem"),
            "unresolved_references": False, "page_size": "US Letter",
            "pdf_sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", default="tectonic")
    parser.add_argument("--skip-isolated-compile", action="store_true")
    args = parser.parse_args()
    records = json.loads((BUILD / "source_conversion_manifest.json").read_text(encoding="utf-8"))
    if [record["version"] for record in records] != list(range(1, 9)):
        raise ValueError("Rebuild all eight versions before packaging.")
    for record in records:
        version = record["version"]
        source = HERE / record["markdown"]
        if sha(source) != record["markdown_sha256"]:
            raise ValueError("Markdown changed after conversion; rebuild first.")
        tex_path = HERE / record["tex"]
        tex = tex_path.read_text(encoding="utf-8")
        if sha(tex_path) != record["tex_sha256"]:
            raise ValueError("TeX changed after conversion; rebuild first.")
        aux = (BUILD / f"paper_v{version}.aux").read_text(encoding="utf-8")
        log = (BUILD / f"paper_v{version}.log").read_text(encoding="utf-8", errors="replace")
        problems = [line for line in log.splitlines() if re.search(
            r"Overfull|Missing character|undefined|LaTeX Error|Package .* Error", line)]
        if problems:
            raise ValueError(f"Compile problems in v{version}: {problems}")
        record.update(check_pdf(HERE / record["pdf"], aux, tex))
        record["underfull_warning_count"] = log.count("Underfull")
        record["markdown_unchanged"] = True
        required = [(tex_path, record["tex"]), (HERE / "neurips_2026.sty", "neurips_2026.sty")]
        for name in record["figures"]:
            required.append((DOCS / "figures" / f"{name}.pdf", f"figures/{name}.pdf"))
        zip_path = HERE / f"paper_v{version}_overleaf.zip"
        package_readme = (
            f"# Overleaf project: revision {version}\n\n"
            f"Upload this ZIP using New Project > Upload Project. Set `{record['tex']}` "
            "as the main document, select **XeLaTeX**, and recompile. "
            "The bibliography is embedded in the TeX; no .bib file is required.\n\n"
            f"The verified local build has {record['main_pages']} main-text pages and "
            f"{record['total_pages']} pages in total. The workshop long-paper limit is "
            "8 main-text pages, excluding references and appendices. Recheck the page "
            "count after compiling with Overleaf's current TeX Live version.\n\n"
            + ("**This revision exceeds the main-text limit and must be shortened before submission.**\n\n"
               if not record["within_eight_main_pages"] else "")
            + "Use the existing double-blind workshop option. Do not add final, preprint, "
            "nonanonymous, or author details for review. The template's generic anonymous "
            "affiliation/address/email block is intentional.\n\n"
            "Formatting verification does not resolve the ethics, data-use, and release "
            "approvals described in the manuscript. No paper has been submitted.\n\n"
            "Required files:\n\n" + "\n".join(f"- `{name}`" for _, name in required) + "\n")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for original, name in required:
                archive.write(original, name)
            archive.writestr("README.md", package_readme)
        record["overleaf_zip"] = zip_path.name
        record["overleaf_zip_sha256"] = sha(zip_path)
        record["required_upload_files"] = [name for _, name in required]
        if not args.skip_isolated_compile:
            isolated = CHECK / f"v{version}"
            isolated.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                for name in archive.namelist():
                    if Path(name).is_absolute() or ".." in Path(name).parts:
                        raise ValueError("Unsafe archive member")
                archive.extractall(isolated)
            result = subprocess.run([args.tectonic, "--only-cached", "--keep-logs", record["tex"]],
                                    cwd=isolated, text=True, capture_output=True,
                                    encoding="utf-8", errors="replace")
            if result.returncode:
                raise RuntimeError(f"Isolated ZIP v{version} compile failed: {result.stdout}\n{result.stderr}")
            isolated_reader = PdfReader(isolated / record["pdf"])
            if len(isolated_reader.pages) != record["total_pages"]:
                raise ValueError("ZIP build pagination differs")
            local_reader = PdfReader(HERE / record["pdf"])
            for original_page, isolated_page in zip(local_reader.pages, isolated_reader.pages):
                if original_page.extract_text() != isolated_page.extract_text():
                    raise ValueError(f"ZIP output text differs in v{version}")
            record["isolated_zip_compile_verified"] = True
        print(f"Verified v{version}: {record['main_pages']} main / {record['total_pages']} total pages; "
              f"{zip_path.name}", flush=True)
    manifest = {"style": "NeurIPS 2026 dblblindworkshop", "body_font_and_margins_unchanged": True,
                "compiler": "Tectonic 0.17.0 (XeTeX); choose XeLaTeX in Overleaf",
                "pandoc": "3.11", "workshop_cfp": "https://physworld-org.github.io/physworld.github.io/cfp/",
                "verification_date": "2026-09-10", "all_markdown_inputs_preserved": True,
                "submission_performed": False, "versions": records}
    (HERE / "submission_build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
