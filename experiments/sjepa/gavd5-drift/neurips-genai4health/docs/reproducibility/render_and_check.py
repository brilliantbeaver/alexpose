"""Render and check publication PDFs using Poppler commands and Pillow.

No PyMuPDF dependency or software installation is required. Automated checks
complement visual inspection; they do not establish scientific validity or
permission to submit or release accompanying data.
"""

from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


DOCS = Path(__file__).resolve().parents[1]
OUT = DOCS / "review/pdf_qa"
LIMITS = {"genai4health_paper_draft": 5, "genai4health_extended_abstract": 2}
FORBIDDEN = ("alexm", "brilliantbeaver", "alexpose_root", "c:\\users", "c:/users", "/users/pmui", "pmui@")


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout


def inspect(stem: str, limit: int) -> dict:
    pdf = DOCS / f"{stem}.pdf"
    tex = DOCS / f"{stem}.tex"
    log_path = DOCS / f"{stem}.log"
    assert pdf.is_file() and tex.is_file() and log_path.is_file(), f"Build {stem} with --keep-logs first."
    info = {}
    for line in command(["pdfinfo", str(pdf)]).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip().lower()] = value.strip()
    info.setdefault("author", "")
    pages = int(info["pages"])
    all_text = command(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"])
    page_texts = all_text.split("\f")
    if not page_texts[-1].strip():
        page_texts.pop()
    assert len(page_texts) == pages

    refs_page, main_pages = None, None
    for index, page_text in enumerate(page_texts):
        lines = page_text.splitlines()
        for line_index, line in enumerate(lines):
            # Anonymous NeurIPS pages prefix headings with review line numbers.
            if re.fullmatch(r"\s*(?:\d+\s+)?References\s*", line):
                refs_page = index + 1
                earlier_content = any(text.strip() for text in lines[:line_index])
                main_pages = refs_page if earlier_content else refs_page - 1
                break
        if refs_page is not None:
            break

    document = ET.fromstring(command(["pdftotext", "-bbox-layout", "-enc", "UTF-8", str(pdf), "-"]))
    outside = []
    page_nodes = [node for node in document.iter() if node.tag.rsplit("}", 1)[-1] == "page"]
    assert len(page_nodes) == pages
    for index, page in enumerate(page_nodes):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        for word in page.iter():
            if word.tag.rsplit("}", 1)[-1] != "word":
                continue
            x0, y0, x1, y1 = (float(word.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax"))
            if x0 < -0.1 or y0 < -0.1 or x1 > width + 0.1 or y1 > height + 0.1:
                outside.append({"page": index + 1, "text": word.text, "bounds": [x0, y0, x1, y1]})

    fonts = []
    for line in command(["pdffonts", str(pdf)]).splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 8:
            fonts.append({"name": parts[0], "embedded": parts[-5] == "yes"})
    assert fonts, "No PDF text fonts found."

    log = log_path.read_text(encoding="utf-8", errors="replace")
    overfull = [line for line in log.splitlines() if re.search(r"Overfull \\[hv]box", line)]
    underfull = [line for line in log.splitlines() if re.search(r"Underfull \\[hv]box", line)]
    missing_glyphs = [line for line in log.splitlines() if "Missing character:" in line]
    unresolved = [line for line in log.splitlines() if re.search(
        r"(?:Citation|Reference).*undefined|There were undefined (?:references|citations)", line)]
    searchable = (all_text + "\n" + json.dumps(info)).lower()
    anonymity_hits = [term for term in FORBIDDEN if term in searchable]
    unresolved_markers = all_text.count("[?]") + len(re.findall(r"(?<!\?)\?\?(?!\?)", all_text))

    folder = OUT / stem
    folder.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for index in range(1, pages + 1):
        prefix = folder / f"page_{index:02d}"
        command(["pdftoppm", "-f", str(index), "-singlefile", "-r", "122", "-png", str(pdf), str(prefix)])
        with Image.open(prefix.with_suffix(".png")) as rendered:
            thumb = rendered.convert("RGB")
            thumb.thumbnail((306, 420))
        canvas = Image.new("RGB", (326, 450), "#E5E9ED")
        canvas.paste(thumb, ((326 - thumb.width) // 2, 12))
        ImageDraw.Draw(canvas).text((12, 432), f"Page {index}", fill="black")
        thumbs.append(canvas)
    # Remove only stale page previews generated under this exact naming convention.
    stale_previews = []
    for path in folder.glob("page_*.png"):
        match = re.fullmatch(r"page_(\d+)\.png", path.name)
        if match and int(match.group(1)) > pages:
            stale_previews.append(path.name)
            path.unlink()
    (folder / "extracted_text.txt").write_text(all_text, encoding="utf-8")
    columns = min(3, pages)
    montage = Image.new("RGB", (columns * 326, ((pages + columns - 1) // columns) * 450), "white")
    for index, thumb in enumerate(thumbs):
        montage.paste(thumb, ((index % columns) * 326, (index // columns) * 450))
    montage.save(folder / "contact_sheet.png")

    checks = {
        "reference_boundary_found": main_pages is not None,
        "within_main_text_limit": main_pages is not None and main_pages <= limit,
        "empty_author_metadata": not info["author"],
        "no_local_identifiers": not anonymity_hits,
        "all_text_inside_page": not outside,
        "all_fonts_embedded": all(font["embedded"] for font in fonts),
        "no_overfull_boxes": not overfull,
        "no_missing_glyphs": not missing_glyphs and "\ufffd" not in all_text,
        "resolved_references_and_citations": not unresolved and unresolved_markers == 0,
    }
    return {
        "pdf": pdf.name, "pages": pages, "references_start_page": refs_page,
        "main_text_pages": main_pages, "main_text_limit": limit, "metadata": info,
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "canonical_tex_sha256": hashlib.sha256(tex.read_bytes()).hexdigest(),
        "outside_page_text": outside, "anonymity_string_hits": anonymity_hits,
        "fonts": fonts, "overfull_boxes": overfull, "underfull_boxes": underfull,
        "missing_glyph_warnings": missing_glyphs,
        "replacement_glyphs": all_text.count("\ufffd"),
        "unresolved_citation_markers": unresolved_markers, "unresolved_warnings": unresolved,
        "checks": checks, "errors": [key for key, passed in checks.items() if not passed],
        "stale_generated_previews_removed": stale_previews,
        "review_scope": "Automated layout checks and rendered previews; no clinical, ethics, permission, or acceptance determination.",
    }


if __name__ == "__main__":
    for executable in ("pdfinfo", "pdftotext", "pdftoppm", "pdffonts"):
        assert shutil.which(executable), f"Required external command unavailable: {executable}"
    OUT.mkdir(parents=True, exist_ok=True)
    results = [inspect(stem, limit) for stem, limit in LIMITS.items()]
    (OUT / "qa_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{key: record[key] for key in (
        "pdf", "pages", "references_start_page", "main_text_pages", "main_text_limit", "errors")}
        for record in results], indent=2))
    if any(record["errors"] for record in results):
        raise SystemExit("Publication QA failed; inspect review/pdf_qa/qa_summary.json before packaging.")
