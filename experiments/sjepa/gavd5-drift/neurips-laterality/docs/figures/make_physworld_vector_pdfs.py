"""Convert the four PhysWorld SVG figures to vector PDF and verify them.

Use the bundled Codex Python (includes pypdf), for example:
  & "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" make_physworld_vector_pdfs.py

Chrome or Edge prints an exact-size, zero-margin HTML wrapper. Text and SVG
paths remain PDF vectors. Poppler renders each resulting PDF independently.
The script never edits the SVGs, their existing PNG previews, or manuscripts.
It refreshes only the four corresponding PDF files, *_pdf_preview.png, and
physworld_pdf_validation.json in this directory. Temporary wrappers and browser
profiles are confined to a newly created subdirectory of tmp/pdfs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from html import escape
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import ContentStream

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
RUNTIME = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
ASSETS = (
    "training_pipeline_compact",
    "training_pipeline",
    "reflection_and_target",
    "learning_results",
)
EXPECTED_TEXT = {
    "training_pipeline_compact": ("Student encoder", "Teacher encoder", "SYNTHETIC DEMONSTRATION ONLY"),
    "training_pipeline": ("Student encoder", "Teacher encoder", "Signed movement contrast"),
    "reflection_and_target": ("Original X", "Reflected MX", "Token equivariance"),
    "learning_results": ("Initial", "375 / 375", "Connected region"),
    "pose_landmarks_and_laterality": ("33 landmarks", "12 gait landmarks", "10 target landmarks"),
}


def run_hidden(command: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, **options)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command[0]}\n{result.stdout}\n{result.stderr}")
    return result


def svg_info(path: Path) -> tuple[str, float, float, str]:
    original = path.read_text(encoding="utf-8")
    root = ET.fromstring(original)
    dimensions = [float(x) for x in root.attrib["viewBox"].split()]
    assert len(dimensions) == 4 and dimensions[:2] == [0.0, 0.0]
    title_node = root.find("{http://www.w3.org/2000/svg}title")
    title = title_node.text if title_node is not None else path.stem
    return original, dimensions[2], dimensions[3], title or path.stem


def inspect_resources(reader, resources, seen=None):
    """Count raster images, fonts, and vector/text operators, including Forms."""
    seen = set() if seen is None else seen
    images, fonts, operations = [], set(), Counter()
    if not resources:
        return images, fonts, operations
    resources = resources.get_object()
    for name, font_ref in resources.get("/Font", {}).items():
        font = font_ref.get_object()
        fonts.add(str(font.get("/BaseFont", name)))
    for name, ref in resources.get("/XObject", {}).items():
        obj = ref.get_object()
        object_id = (getattr(ref, "idnum", None), getattr(ref, "generation", None))
        if object_id != (None, None) and object_id in seen:
            continue
        seen.add(object_id)
        if obj.get("/Subtype") == "/Image":
            images.append({"name": str(name), "width": obj.get("/Width"), "height": obj.get("/Height")})
        elif obj.get("/Subtype") == "/Form":
            nested = ContentStream(obj, reader)
            operations.update(op.decode("ascii") for _, op in nested.operations)
            inner_images, inner_fonts, inner_ops = inspect_resources(reader, obj.get("/Resources"), seen)
            images.extend(inner_images)
            fonts.update(inner_fonts)
            operations.update(inner_ops)
    return images, fonts, operations


def validate_pdf(path: Path, width: float, height: float, stem: str) -> dict:
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise AssertionError(f"{path.name}: expected one page, found {len(reader.pages)}")
    page = reader.pages[0]
    actual = [float(page.mediabox.width), float(page.mediabox.height)]
    expected = [width * 0.75, height * 0.75]
    if any(abs(a-b) > 0.8 for a, b in zip(actual, expected)):
        raise AssertionError(f"{path.name}: page size {actual} differs from {expected}")
    extracted = page.extract_text() or ""
    normalized = re.sub(r"\s+", " ", extracted)
    expected_text = EXPECTED_TEXT.get(stem.removesuffix("_print"))
    if expected_text is None:
        expected_text = (svg_info(HERE / f"{stem}.svg")[3],)
    if stem in ("training_pipeline_compact", "training_pipeline_compact_print"):
        expected_text = ("Student", "Teacher", "SYNTHETIC ONLY")
    for phrase in expected_text:
        if phrase not in normalized:
            raise AssertionError(f"{path.name}: searchable text missing {phrase!r}")
    images, fonts, operations = inspect_resources(reader, page.get("/Resources"))
    contents = page.get_contents()
    if contents is not None:
        operations.update(op.decode("ascii") for _, op in ContentStream(contents, reader).operations)
    vector_count = sum(operations[k] for k in ("m", "l", "c", "v", "y", "re"))
    text_count = sum(operations[k] for k in ("Tj", "TJ", "'", '"'))
    if images or vector_count < 10 or text_count < 10:
        raise AssertionError(f"{path.name}: vector/text validation failed: images={images}, vectors={vector_count}, text={text_count}")
    return {
        "pages": 1,
        "svg_pixels": [width, height],
        "expected_points_at_96_dpi": expected,
        "page_points": actual,
        "text_characters": len(extracted),
        "searchable_text_checks": list(expected_text),
        "fonts": sorted(fonts),
        "raster_image_xobjects": images,
        "vector_path_operations": vector_count,
        "text_show_operations": text_count,
        "pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--pdftoppm", type=Path, default=RUNTIME / "native/poppler/Library/bin/pdftoppm.exe")
    parser.add_argument("--print-variants", action="store_true", help="Convert only the additional *_print.svg figures.")
    parser.add_argument("--assets", nargs="+", help="Convert only these explicitly named SVG stems.")
    parser.add_argument("--report-name", help="Write validation to this JSON basename in the figure directory.")
    parser.add_argument("--publish-preview", action="store_true", help="Also publish each new PDF preview as ASSET.png; explicit replacement of that PNG.")
    args = parser.parse_args()
    candidates = [args.browser] if args.browser else [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ]
    browser = next((p for p in candidates if p and p.is_file()), None)
    if not browser or not args.pdftoppm.is_file():
        raise FileNotFoundError("Supply an installed Chrome/Edge and Poppler pdftoppm; no input has been modified.")
    temp_root = WORKSPACE / "tmp/pdfs"
    temp_root.mkdir(parents=True, exist_ok=True)
    results = {}
    source_hashes = {}
    selected_assets = tuple(args.assets) if args.assets else tuple(f"{stem}_print" for stem in ASSETS) if args.print_variants else ASSETS
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", stem) for stem in selected_assets):
        raise ValueError("Asset names must be simple SVG basenames without suffixes or directories.")
    for stem in selected_assets:
        for suffix in (".svg", ".png"):
            if suffix == ".png" and args.publish_preview:
                continue
            source = HERE / f"{stem}{suffix}"
            if source.is_file():
                source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        source, width, height, title = svg_info(HERE / f"{stem}.svg")
        # TemporaryDirectory removes only its own validated, unique directory.
        with tempfile.TemporaryDirectory(prefix="physworld_svg_", dir=temp_root, ignore_cleanup_errors=True) as temporary:
            staging = Path(temporary)
            wrapper = staging / f"{stem}.html"
            wrapper.write_text(f'''<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>@page {{size: {width}px {height}px; margin: 0;}}
html, body {{margin: 0; padding: 0; width: {width}px; height: {height}px;}}
body {{print-color-adjust: exact; -webkit-print-color-adjust: exact;}}
svg {{display: block; width: {width}px; height: {height}px;}}</style></head>
<body>{source}</body></html>''', encoding="utf-8")
            destination = HERE / f"{stem}.pdf"
            command = [str(browser), "--headless=new", "--disable-gpu", "--no-first-run",
                       "--no-pdf-header-footer", f"--user-data-dir={staging / 'browser-profile'}",
                       f"--print-to-pdf={destination}", wrapper.as_uri()]
            run_hidden(command)
        if not destination.is_file():
            raise RuntimeError(f"Browser did not produce {destination}")
        results[stem] = validate_pdf(destination, width, height, stem)
        preview = HERE / f"{stem}_pdf_preview"
        run_hidden([str(args.pdftoppm), "-r", "96", "-singlefile", "-png", str(destination), str(preview)])
        if args.publish_preview:
            shutil.copyfile(HERE / f"{stem}_pdf_preview.png", HERE / f"{stem}.png")
        results[stem]["preview"] = f"{stem}_pdf_preview.png"
        print(f"{stem}.pdf: {results[stem]['page_points']} pt; {results[stem]['vector_path_operations']} vector operations; {results[stem]['text_show_operations']} text operations; zero raster images")
    for name, digest in source_hashes.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Original SVG/PNG was altered during conversion: {name}")
    report = {
        "method": "Headless Chromium print with exact-size zero-margin CSS page; independent Poppler rendering",
        "browser": str(browser),
        "pdftoppm": str(args.pdftoppm),
        "original_svg_png_sha256_unchanged": source_hashes,
        "assets": results,
    }
    report_name = args.report_name or ("physworld_custom_pdf_validation.json" if args.assets else "physworld_print_pdf_validation.json" if args.print_variants else "physworld_pdf_validation.json")
    if Path(report_name).name != report_name or not report_name.endswith(".json"):
        raise ValueError("The validation report must be a JSON basename in this figure directory.")
    (HERE / report_name).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
