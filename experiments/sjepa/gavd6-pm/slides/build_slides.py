#!/usr/bin/env python3
"""Build and validate the S-JEPA gait deck as PowerPoint, standalone HTML, and PDF.

The file names deliberately carry no generation number. This deck describes how the
study evolved across several GAVD generations, so pinning it to any one of them would
both misdescribe the content and go stale at the next iteration.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path


SLIDE_DIR = Path(__file__).resolve().parent
REPO_DIR = SLIDE_DIR.parent
SOURCE = SLIDE_DIR / "GAVD_SJEPA_Gait_Tutorial.md"
HTML_OUTPUT = SLIDE_DIR / "GAVD_SJEPA_Gait_Tutorial.html"
PPTX_OUTPUT = SLIDE_DIR / "GAVD_SJEPA_Gait_Tutorial.pptx"
PDF_OUTPUT = SLIDE_DIR / "GAVD_SJEPA_Gait_Tutorial.pdf"
PDF_PREAMBLE = SLIDE_DIR / "beamer_preamble.tex"
PDF_FILTER = SLIDE_DIR / "beamer_slide_fit.lua"
TEMPLATE = SLIDE_DIR / "html" / "template.html"
STYLESHEET = SLIDE_DIR / "html" / "slides.css"
RUNTIME = SLIDE_DIR / "html" / "slides.js"
EXPECTED_MAIN_SLIDES = 17
# The booked conference slot is ten minutes including questions. Pace depends on delivery,
# so this is reported and never enforced: failing a build on it would block a deck that is
# correct but dense, which is a judgement call for the presenter rather than the builder.
TALK_MINUTES = 10
QUESTION_MINUTES = 2
COMFORTABLE_SECONDS_PER_SLIDE = 40
PPTX_THEME_REPLACEMENTS = {
    '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">':
        '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="GAVD S-JEPA Theme">',
    '<a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>':
        '<a:dk1><a:srgbClr val="17324D"/></a:dk1>',
    '<a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>':
        '<a:lt1><a:srgbClr val="FBFAF7"/></a:lt1>',
    '<a:dk2><a:srgbClr val="1F497D"/></a:dk2>':
        '<a:dk2><a:srgbClr val="244D70"/></a:dk2>',
    '<a:lt2><a:srgbClr val="EEECE1"/></a:lt2>':
        '<a:lt2><a:srgbClr val="EEF3F6"/></a:lt2>',
    '<a:accent1><a:srgbClr val="4F81BD"/></a:accent1>':
        '<a:accent1><a:srgbClr val="3977A8"/></a:accent1>',
    '<a:accent2><a:srgbClr val="C0504D"/></a:accent2>':
        '<a:accent2><a:srgbClr val="D97745"/></a:accent2>',
    '<a:accent3><a:srgbClr val="9BBB59"/></a:accent3>':
        '<a:accent3><a:srgbClr val="3D8B7D"/></a:accent3>',
    '<a:accent4><a:srgbClr val="8064A2"/></a:accent4>':
        '<a:accent4><a:srgbClr val="7562A8"/></a:accent4>',
    '<a:accent5><a:srgbClr val="4BACC6"/></a:accent5>':
        '<a:accent5><a:srgbClr val="D9A441"/></a:accent5>',
    '<a:accent6><a:srgbClr val="F79646"/></a:accent6>':
        '<a:accent6><a:srgbClr val="B94E48"/></a:accent6>',
    '<a:hlink><a:srgbClr val="0000FF"/></a:hlink>':
        '<a:hlink><a:srgbClr val="1D6395"/></a:hlink>',
    '<a:folHlink><a:srgbClr val="800080"/></a:folHlink>':
        '<a:folHlink><a:srgbClr val="7562A8"/></a:folHlink>',
}


class SlideHTMLAudit(HTMLParser):
    """Collect structural and offline-resource checks from generated HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections = 0
        self.section_ids: list[str] = []
        self.images = 0
        self.image_errors: list[str] = []
        self.resource_errors: list[str] = []
        self.external_links = 0

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key: value or "" for key, value in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = self._attrs(attrs)
        if tag == "section":
            self.sections += 1
            if values.get("id"):
                self.section_ids.append(values["id"])
        elif tag == "img":
            self.images += 1
            src = values.get("src", "")
            alt = values.get("alt", "").strip() or values.get("aria-label", "").strip()
            if not src.startswith("data:"):
                self.image_errors.append(f"image is not embedded: {src or '<missing src>'}")
            if not alt:
                self.image_errors.append(f"image has no alternative text: {src[:80]}")
        elif tag == "script" and values.get("src"):
            if not values["src"].startswith("data:"):
                self.resource_errors.append(f"external script resource: {values['src']}")
        elif tag == "link" and "stylesheet" in values.get("rel", ""):
            if not values.get("href", "").startswith("data:"):
                self.resource_errors.append(f"external stylesheet resource: {values.get('href', '')}")
        elif tag == "a" and values.get("href", "").startswith(("http://", "https://")):
            self.external_links += 1


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=SLIDE_DIR, check=True)


def source_slide_counts(markdown: str) -> tuple[int, int, int]:
    headings = re.findall(r"^# (.+)$", markdown, flags=re.MULTILINE)
    if not headings:
        fail("The Markdown source contains no level-one slide headings.")
    first_appendix = next(
        (index for index, title in enumerate(headings) if title.startswith("Appendix:")),
        len(headings),
    )
    main = 1 + first_appendix
    appendix = len(headings) - first_appendix
    return main, appendix, 1 + len(headings)


def timing_note(main_slides: int) -> str:
    """Describe the pace the main-slide count implies for the booked slot."""
    speaking_minutes = TALK_MINUTES - QUESTION_MINUTES
    seconds_each = speaking_minutes * 60 / main_slides
    pace = "within" if seconds_each >= COMFORTABLE_SECONDS_PER_SLIDE else "over"
    return (
        f"Timing: {main_slides} main slides in {speaking_minutes} speaking minutes of a "
        f"{TALK_MINUTES}-minute slot is about {seconds_each:.0f} seconds each, {pace} the "
        f"{COMFORTABLE_SECONDS_PER_SLIDE}-second guide. Advisory only."
    )


def validate_sources() -> tuple[int, int, int]:
    required = [SOURCE, TEMPLATE, STYLESHEET, RUNTIME, PDF_PREAMBLE, PDF_FILTER]
    missing = [str(path.relative_to(REPO_DIR)) for path in required if not path.exists()]
    if missing:
        fail("Missing slide source files: " + ", ".join(missing))

    for path in required:
        text = path.read_text(encoding="utf-8")
        if "\u2014" in text:
            fail(f"Em dash found in {path.relative_to(REPO_DIR)}")

    markdown = SOURCE.read_text(encoding="utf-8")
    main, appendix, total = source_slide_counts(markdown)
    if main != EXPECTED_MAIN_SLIDES:
        fail(f"Expected {EXPECTED_MAIN_SLIDES} main slides including the title, found {main}.")

    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)")
    image_matches = image_pattern.findall(markdown)
    if not image_matches:
        fail("The slide source contains no images.")
    for alt, raw_path in image_matches:
        if not alt.strip():
            fail(f"Image has empty alternative text: {raw_path}")
        if raw_path.startswith(("http://", "https://", "data:")):
            continue
        resolved = (SLIDE_DIR / raw_path).resolve()
        if not resolved.exists():
            fail(f"Missing image referenced by slides: {raw_path}")

    if shutil.which("pandoc") is None:
        fail("Pandoc is required to build the slides.")
    if shutil.which("node") is None:
        fail("Node is required to validate the HTML runtime.")
    run(["node", "--check", str(RUNTIME.relative_to(SLIDE_DIR))])
    return main, appendix, total


def build_html() -> None:
    resource_path = os.pathsep.join([".", ".."])
    run(
        [
            "pandoc",
            SOURCE.name,
            "--from=markdown",
            "--to=dzslides",
            "--standalone",
            "--slide-level=1",
            f"--template={TEMPLATE.relative_to(SLIDE_DIR)}",
            f"--css={STYLESHEET.relative_to(SLIDE_DIR)}",
            f"--resource-path={resource_path}",
            "--embed-resources",
            "--mathml",
            "-o",
            HTML_OUTPUT.name,
        ]
    )

    html = HTML_OUTPUT.read_text(encoding="utf-8")
    html = html.replace('<section class="title">', '<section id="slide-1" class="title">', 1)
    HTML_OUTPUT.write_text(html, encoding="utf-8")


def build_pptx() -> None:
    resource_path = os.pathsep.join([".", ".."])
    run(
        [
            "pandoc",
            SOURCE.name,
            "--from=markdown",
            "--to=pptx",
            "--standalone",
            "--slide-level=1",
            f"--resource-path={resource_path}",
            "-o",
            PPTX_OUTPUT.name,
        ]
    )
    apply_pptx_theme()


def build_pdf() -> None:
    """Render the same source to a slide-per-page PDF through Beamer.

    Beamer gives one page per slide at 16:9, which keeps the PDF readable as a deck
    rather than as a wall of prose. Speaker notes stay hidden, which is Beamer's default,
    so the PDF matches what an audience sees rather than what the presenter reads.
    """
    resource_path = os.pathsep.join([".", ".."])
    run(
        [
            "pandoc",
            SOURCE.name,
            "--from=markdown",
            "--to=beamer",
            "--standalone",
            "--slide-level=1",
            f"--resource-path={resource_path}",
            f"--lua-filter={PDF_FILTER.relative_to(SLIDE_DIR)}",
            f"--include-in-header={PDF_PREAMBLE.relative_to(SLIDE_DIR)}",
            "--pdf-engine=tectonic",
            "-V",
            "aspectratio=169",
            "-V",
            "fontsize=9pt",
            "-V",
            "theme=default",
            "-V",
            "colortheme=seahorse",
            "-V",
            "linkcolor=blue",
            "-o",
            PDF_OUTPUT.name,
        ]
    )


def apply_pptx_theme() -> None:
    """Apply portable fonts and the project palette without changing slide content."""

    temporary = PPTX_OUTPUT.with_name(PPTX_OUTPUT.stem + ".styled.tmp.pptx")
    try:
        with zipfile.ZipFile(PPTX_OUTPUT, "r") as source, zipfile.ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "ppt/theme/theme1.xml":
                    theme = data.decode("utf-8")
                    for old, new in PPTX_THEME_REPLACEMENTS.items():
                        if old not in theme:
                            fail(f"Expected PowerPoint theme fragment was not found: {old[:80]}")
                        theme = theme.replace(old, new, 1)
                    if '<a:latin typeface="Calibri"/>' not in theme:
                        fail("The generated PowerPoint theme does not contain the expected Calibri font entry.")
                    theme = theme.replace('<a:latin typeface="Calibri"/>', '<a:latin typeface="Arial"/>')
                    data = theme.encode("utf-8")
                target.writestr(item, data)
        os.replace(temporary, PPTX_OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_html(expected_slides: int) -> SlideHTMLAudit:
    if not HTML_OUTPUT.exists():
        fail(f"Missing generated HTML: {HTML_OUTPUT.name}")
    text = HTML_OUTPUT.read_text(encoding="utf-8")
    audit = SlideHTMLAudit()
    audit.feed(text)
    if audit.sections != expected_slides:
        fail(f"HTML has {audit.sections} slides; expected {expected_slides}.")
    if len(audit.section_ids) != expected_slides:
        fail("Every generated HTML slide must have a static ID.")
    if len(set(audit.section_ids)) != expected_slides:
        fail("Generated HTML slide IDs are not unique.")
    if audit.images == 0:
        fail("Generated HTML contains no images.")
    errors = audit.image_errors + audit.resource_errors
    if errors:
        fail("HTML offline audit failed:\n- " + "\n- ".join(errors))
    if re.search(r"(?:src|href)=[\"']https?://", text):
        links_only = re.sub(r"<a\b[^>]*href=[\"']https?://[^>]*>", "", text)
        if re.search(r"(?:src|href)=[\"']https?://", links_only):
            fail("Generated HTML contains a remote runtime resource.")
    if re.search(r"url\(\s*[\"']?https?://", text, flags=re.IGNORECASE):
        fail("Generated HTML CSS contains a remote URL.")
    return audit


def validate_pptx(expected_slides: int) -> tuple[int, int]:
    if not PPTX_OUTPUT.exists():
        fail(f"Missing generated PowerPoint: {PPTX_OUTPUT.name}")
    with zipfile.ZipFile(PPTX_OUTPUT) as archive:
        bad = archive.testzip()
        if bad:
            fail(f"Corrupt PowerPoint member: {bad}")
        names = archive.namelist()
        theme = archive.read("ppt/theme/theme1.xml").decode("utf-8")
        if "GAVD S-JEPA Theme" not in theme or '<a:latin typeface="Arial"/>' not in theme:
            fail("The generated PowerPoint does not contain the expected GAVD S-JEPA theme.")
        slides = [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
        notes = [name for name in names if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)]
        if len(slides) != expected_slides:
            fail(f"PowerPoint has {len(slides)} slides; expected {expected_slides}.")
        namespaces = {
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        }
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        slide_size = presentation.find("p:sldSz", namespaces)
        if slide_size is None:
            fail("PowerPoint slide dimensions are missing.")
        width = int(slide_size.attrib["cx"])
        height = int(slide_size.attrib["cy"])
        if width * 9 != height * 16:
            fail(f"PowerPoint is not 16:9: {width} by {height} EMU.")
        for slide_name in slides:
            slide_root = ET.fromstring(archive.read(slide_name))
            for transform in (element for element in slide_root.iter() if element.tag.endswith("}xfrm")):
                offset = transform.find("a:off", namespaces)
                extent = transform.find("a:ext", namespaces)
                if offset is None or extent is None:
                    continue
                x = int(offset.attrib["x"])
                y = int(offset.attrib["y"])
                shape_width = int(extent.attrib["cx"])
                shape_height = int(extent.attrib["cy"])
                if x < 0 or y < 0 or x + shape_width > width or y + shape_height > height:
                    fail(f"PowerPoint shape exceeds slide bounds in {slide_name}.")
    return len(slides), len(notes)


def pdf_page_count() -> int | None:
    """Return the PDF page count, or None when poppler is not installed.

    The page check is a nice-to-have rather than a build requirement, so a machine
    without poppler still gets a PDF instead of a failure.
    """
    if shutil.which("pdfinfo") is None:
        return None
    output = subprocess.run(
        ["pdfinfo", PDF_OUTPUT.name],
        cwd=SLIDE_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
    if not match:
        fail("Could not read a page count from the generated PDF.")
    return int(match.group(1))


def validate_pdf(expected_slides: int) -> int | None:
    if not PDF_OUTPUT.exists():
        fail(f"Missing generated PDF: {PDF_OUTPUT.name}")
    with PDF_OUTPUT.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            fail("The generated PDF does not start with a PDF header.")
    pages = pdf_page_count()
    if pages is not None and pages != expected_slides:
        fail(f"PDF has {pages} pages; expected one per slide, so {expected_slides}.")
    return pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("all", "html", "pptx", "pdf"),
        default="all",
        help="output to build or validate",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate existing outputs without rebuilding them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        main_slides, appendix_slides, total_slides = validate_sources()
        if not args.check_only:
            if args.format in ("all", "html"):
                build_html()
            if args.format in ("all", "pptx"):
                build_pptx()
            if args.format in ("all", "pdf"):
                build_pdf()

        html_audit = None
        pptx_summary = None
        pdf_pages = None
        if args.format in ("all", "html"):
            html_audit = validate_html(total_slides)
        if args.format in ("all", "pptx"):
            pptx_summary = validate_pptx(total_slides)
        if args.format in ("all", "pdf"):
            pdf_pages = validate_pdf(total_slides)

        print(f"Source: {main_slides} main slides, {appendix_slides} appendix slides, {total_slides} total")
        print(timing_note(main_slides))
        if html_audit:
            size = HTML_OUTPUT.stat().st_size
            print(
                f"HTML: {html_audit.sections} slides, {html_audit.images} embedded images, "
                f"{html_audit.external_links} optional citation links, {size:,} bytes"
            )
        if pptx_summary:
            slide_count, note_count = pptx_summary
            size = PPTX_OUTPUT.stat().st_size
            print(f"PowerPoint: {slide_count} slides, {note_count} note pages, {size:,} bytes")
        if pdf_pages is not None or (args.format in ("all", "pdf") and PDF_OUTPUT.exists()):
            size = PDF_OUTPUT.stat().st_size
            pages = f"{pdf_pages} pages" if pdf_pages is not None else "page count not checked"
            print(f"PDF: {pages}, {size:,} bytes")
        print("Validation passed.")
        return 0
    except (RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
