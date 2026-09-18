#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-docx>=1.1.2,<2", "lxml>=5,<7"]
# ///
"""Build the full progress report as a styled, editable Microsoft Word document.

Run from any directory:
    uv run --script scripts/scripts_build_progress_docx.py

Requires pandoc and rsvg-convert on PATH. Dependencies are isolated by uv; the
notebook environment is not changed. All source content comes from Markdown.
Equations become native Office Math, tables remain editable, and each diagram
contains its original SVG plus a 3x PNG fallback for older Office readers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs/08-0913-PROGRESS.MD"
NAVY, TEAL, INK, MUTED = "142B45", "007E87", "253745", "536675"
PAGE_WIDTH = 7.15
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main",
}


def command(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, capture_output=True, **kwargs)


def nodes(value, kind: str):
    if isinstance(value, dict):
        if value.get("t") == kind:
            yield value
        for child in value.values():
            yield from nodes(child, kind)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child, kind)


def inline_text(value) -> str:
    if isinstance(value, list):
        return "".join(inline_text(x) for x in value)
    if not isinstance(value, dict):
        return ""
    kind, c = value.get("t"), value.get("c")
    if kind in {"Str", "MetaString"}:
        return c
    if kind in {"Space", "SoftBreak", "LineBreak"}:
        return " "
    if kind in {"Code", "Math"}:
        return c[-1]
    if kind in {"Link", "Image"}:
        return inline_text(c[1])
    if kind == "Header":
        return inline_text(c[-1])
    return inline_text(c)


def set_font(style, name="Arial", size=11, color=INK, bold=False):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor.from_string(color)
    style.font.bold = bold
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), name)


def add_style(doc, name, size, color=INK, bold=False, based="Normal"):
    style = doc.styles[name] if name in doc.styles else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    style.base_style = doc.styles[based]
    set_font(style, size=size, color=color, bold=bold)
    return style


def add_field(paragraph, instruction, cached="1"):
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    paragraph.add_run()._r.append(begin)
    code = OxmlElement("w:instrText")
    code.set(qn("xml:space"), "preserve")
    code.text = f" {instruction} "
    paragraph.add_run()._r.append(code)
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    paragraph.add_run()._r.append(separate)
    paragraph.add_run(cached)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    paragraph.add_run()._r.append(end)


def border(element, name, value="single", color="D3DFE6", size="6"):
    node = OxmlElement(f"w:{name}")
    node.set(qn("w:val"), value)
    node.set(qn("w:color"), color)
    node.set(qn("w:sz"), size)
    node.set(qn("w:space"), "8")
    element.append(node)


def build_reference(path: Path):
    default = command("pandoc", "--print-default-data-file", "reference.docx").stdout
    path.write_bytes(default)
    doc = Document(path)
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.left_margin = section.right_margin = Inches(0.675)
    section.top_margin, section.bottom_margin = Inches(0.7), Inches(0.7)
    section.header_distance = section.footer_distance = Inches(0.3)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    set_font(normal)
    normal.paragraph_format.line_spacing = 1.13
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.widow_control = True
    for name in ("Body Text", "First Paragraph", "Compact"):
        style = doc.styles[name]
        style.base_style = normal
        set_font(style)
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.line_spacing = 1.13
        style.paragraph_format.first_line_indent = Inches(0)
    for level, size in ((1, 20), (2, 14), (3, 12)):
        style = doc.styles[f"Heading {level}"]
        set_font(style, size=size, color=NAVY, bold=True)
        pf = style.paragraph_format
        pf.space_before, pf.space_after = Pt(20 if level == 1 else 12), Pt(7)
        pf.line_spacing, pf.keep_with_next = 1.07, True
        pf.keep_together, pf.page_break_before = True, False
    set_font(doc.styles["Title"], size=32, color=NAVY, bold=True)
    doc.styles["Title"].paragraph_format.space_before = Pt(28)
    doc.styles["Title"].paragraph_format.space_after = Pt(20)
    doc.styles["Title"].paragraph_format.line_spacing = 1.05
    set_font(doc.styles["Subtitle"], size=13, color=TEAL)
    doc.styles["Subtitle"].paragraph_format.space_after = Pt(24)
    for name in ("Verbatim Char", "Source Code"):
        if name in doc.styles:
            set_font(doc.styles[name], name="Consolas", size=9.3, color=INK)
    if "Hyperlink" in doc.styles:
        set_font(doc.styles["Hyperlink"], size=11, color=TEAL)
        doc.styles["Hyperlink"].font.underline = True
    for name in ("Caption", "Image Caption"):
        if name in doc.styles:
            set_font(doc.styles[name], size=9.5, color=MUTED)
            doc.styles[name].paragraph_format.line_spacing = 1.1
            doc.styles[name].paragraph_format.space_after = Pt(12)
            doc.styles[name].paragraph_format.keep_together = True
    cover_intro = add_style(doc, "Cover Intro", 12, color=MUTED)
    cover_intro.paragraph_format.line_spacing = 1.25
    cover_intro.paragraph_format.space_after = Pt(20)
    add_style(doc, "Cover Kicker", 10, color=TEAL, bold=True)
    table_text = add_style(doc, "Report Table", 9.5)
    table_text.paragraph_format.line_spacing = 1.07
    table_text.paragraph_format.space_after = Pt(0)
    table_text.paragraph_format.keep_with_next = False
    toc = add_style(doc, "Report Contents", 11.5, color=NAVY)
    toc.paragraph_format.space_after = Pt(12)
    toc.paragraph_format.line_spacing = 1.12
    toc.paragraph_format.keep_together = True
    add_style(doc, "Contents Title", 24, color=NAVY, bold=True)
    for name in ("Header", "Footer"):
        add_style(doc, name, 8.5, color=MUTED)

    header = section.header.paragraphs[0]
    header.style = doc.styles["Header"]
    header.paragraph_format.tab_stops.add_tab_stop(Inches(PAGE_WIDTH), WD_ALIGN_PARAGRAPH.RIGHT)
    header.add_run("GAIT GEOMETRY / S-JEPA")
    header.add_run("\t13 SEPTEMBER 2026")
    borders = OxmlElement("w:pBdr")
    border(borders, "bottom")
    header._p.get_or_add_pPr().append(borders)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Research progress  •  ")
    add_field(footer, "PAGE")
    doc.core_properties.title = "Geometry, symmetry, and learned gait representations"
    doc.core_properties.subject = "Research progress through 13 September 2026"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "gait, geometry, symmetry, S-JEPA, source-separated evaluation"
    doc.core_properties.comments = "Generated from docs/08-0913-PROGRESS.MD; native text, tables, equations, and vector diagrams."
    doc.save(path)


def table_widths(table):
    first = table.cell(0, 0).text
    count = len(table.columns)
    widths = {
        "Question": [2.0, 3.05, 2.1],
        "Notebook": [2.0, 5.15],
        "Dataset stage": [2.12, 1.23, 1.15, 1.15, 1.5],
        "Notebook 04 comparison": [3.0, 1.48, 1.0, 1.67],
        "Fold": [0.55, 1.9, 1.85, 2.85],
        "System": [3.0, 1.04, 0.82, 0.95, 1.34],
        "True condition": [1.35, 0.65, 2.57, 2.58],
        "True label": [1.03, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02],
        "Control and its actual input": [3.45, 1.85, 1.85],
        "Evidence": [1.8, 5.35],
    }.get(first, [PAGE_WIDTH / count] * count)
    if len(widths) != count:
        return [PAGE_WIDTH / count] * count
    return widths


def style_tables(doc):
    for table in doc.tables:
        table.autofit = False
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        widths = table_widths(table)
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
        props = table._tbl.tblPr
        edges = OxmlElement("w:tblBorders")
        for name in ("top", "left", "bottom", "right", "insideV"):
            border(edges, name, value="nil")
        border(edges, "insideH", size="4", color="DFE7EC")
        props.append(edges)
        for i, row in enumerate(table.rows):
            tr_pr = row._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:cantSplit"))
            if i == 0:
                tr_pr.append(OxmlElement("w:tblHeader"))
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tc_pr = cell._tc.get_or_add_tcPr()
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), NAVY if i == 0 else ("EFF4F7" if i % 2 else "FFFFFF"))
                tc_pr.append(shading)
                margins = OxmlElement("w:tcMar")
                for name, value in (("top", "95"), ("bottom", "95"), ("left", "105"), ("right", "105")):
                    margin = OxmlElement(f"w:{name}")
                    margin.set(qn("w:w"), value)
                    margin.set(qn("w:type"), "dxa")
                    margins.append(margin)
                tc_pr.append(margins)
                for paragraph in cell.paragraphs:
                    paragraph.style = doc.styles["Report Table"]
                    for run in paragraph.runs:
                        run.font.size = Pt(9 if i == 0 else 9.5)
                        if i == 0:
                            run.font.color.rgb = RGBColor(255, 255, 255)
                            run.font.bold = True
                    # Hyperlink runs are not included in paragraph.runs.
                    for run in paragraph._p.xpath(".//w:hyperlink/w:r"):
                        pr = run.get_or_add_rPr()
                        size = OxmlElement("w:sz")
                        size.set(qn("w:val"), "19")
                        pr.append(size)


def hyperlink(paragraph, label, anchor):
    link = OxmlElement("w:hyperlink")
    link.set(qn("w:anchor"), anchor)
    link.set(qn("w:history"), "1")
    run = OxmlElement("w:r")
    pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), NAVY)
    pr.append(color)
    run.append(pr)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


def style_document(path: Path, title: str):
    doc = Document(path)
    title_paragraph = doc.paragraphs[0]
    title_paragraph.style = doc.styles["Title"]
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    kicker = title_paragraph.insert_paragraph_before("MULTIPLE SCLEROSIS RESEARCH", style="Cover Kicker")
    kicker.paragraph_format.space_before = Pt(55)
    introductory = doc.paragraphs[2]
    introductory.style = doc.styles["Cover Intro"]
    subtitle = introductory.insert_paragraph_before("A step-by-step review of the retained notebook evidence", style="Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.LEFT
    subtitle.paragraph_format.space_after = Pt(28)

    first_heading = next(p for p in doc.paragraphs if p.style.name == "Heading 1")
    toc_title = first_heading.insert_paragraph_before("Contents", style="Contents Title")
    toc_title.paragraph_format.page_break_before = True
    toc_title.paragraph_format.space_before = Pt(10)
    toc_title.paragraph_format.space_after = Pt(16)
    guide = first_heading.insert_paragraph_before("Select a section to move through the report.")
    guide.paragraph_format.space_after = Pt(20)
    for run in guide.runs:
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor.from_string(MUTED)
    headings = [p for p in doc.paragraphs if p.style.name == "Heading 1"]
    for i, heading in enumerate(headings, 1):
        # Stable bookmarks are independent of punctuation in section names.
        anchor = f"report_section_{i}"
        start, end = OxmlElement("w:bookmarkStart"), OxmlElement("w:bookmarkEnd")
        start.set(qn("w:id"), str(1000 + i))
        start.set(qn("w:name"), anchor)
        end.set(qn("w:id"), str(1000 + i))
        heading._p.insert(0, start)
        heading._p.append(end)
        entry = first_heading.insert_paragraph_before(style="Report Contents")
        hyperlink(entry, heading.text, anchor)
    first_heading.paragraph_format.page_break_before = True
    first_heading.paragraph_format.space_before = Pt(0)

    for paragraph in doc.paragraphs:
        if paragraph._p.xpath(".//w:drawing"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.keep_together = True
            paragraph.paragraph_format.space_before = Pt(8)
            paragraph.paragraph_format.space_after = Pt(5)
        if re.match(r"Figure \d+\.", paragraph.text):
            paragraph.style = doc.styles["Image Caption"]
        if paragraph._p.xpath(".//m:oMathPara"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(6)
            paragraph.paragraph_format.space_after = Pt(12)
            paragraph.paragraph_format.keep_together = True
    style_tables(doc)
    settings = doc.settings.element
    compat = settings.find(qn("w:compat"))
    if compat is None:
        compat = OxmlElement("w:compat")
        settings.append(compat)
    setting = OxmlElement("w:compatSetting")
    setting.set(qn("w:name"), "compatibilityMode")
    setting.set(qn("w:uri"), "http://schemas.microsoft.com/office/word")
    setting.set(qn("w:val"), "15")
    compat.append(setting)
    doc.core_properties.title = title
    doc.save(path)


def embed_vectors(path: Path, images: list[Path]):
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    xml = etree.fromstring(files["word/document.xml"])
    rels = etree.fromstring(files["word/_rels/document.xml.rels"])
    content_types = etree.fromstring(files["[Content_Types].xml"])
    if not content_types.xpath("ct:Default[@Extension='svg']", namespaces=NS):
        declaration = etree.SubElement(content_types, f"{{{NS['ct']}}}Default")
        declaration.set("Extension", "svg")
        declaration.set("ContentType", "image/svg+xml")
    drawings = xml.xpath("//w:drawing", namespaces=NS)
    if len(drawings) != len(images):
        raise ValueError(f"Expected {len(images)} drawings; got {len(drawings)}")
    for index, (drawing, source) in enumerate(zip(drawings, images), 1):
        name, rel_id = f"report-vector-{index:02}.svg", f"rIdReportSvg{index}"
        files[f"word/media/{name}"] = source.read_bytes()
        relationship = etree.SubElement(rels, f"{{{NS['rel']}}}Relationship")
        relationship.set("Id", rel_id)
        relationship.set("Type", f"{NS['r']}/image")
        relationship.set("Target", f"media/{name}")
        blip = drawing.xpath(".//a:blip", namespaces=NS)[0]
        ext_list = blip.find(f"{{{NS['a']}}}extLst")
        if ext_list is None:
            ext_list = etree.SubElement(blip, f"{{{NS['a']}}}extLst")
        ext = etree.SubElement(ext_list, f"{{{NS['a']}}}ext")
        ext.set("uri", "{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
        svg = etree.SubElement(ext, f"{{{NS['asvg']}}}svgBlip", nsmap={"asvg": NS["asvg"]})
        svg.set(f"{{{NS['r']}}}embed", rel_id)
    for name, tree in (("word/document.xml", xml), ("word/_rels/document.xml.rels", rels), ("[Content_Types].xml", content_types)):
        files[name] = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def validate(path: Path, ast: dict, images: list[Path]) -> dict:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, "Corrupt ZIP member"
        xml = etree.fromstring(archive.read("word/document.xml"))
        rels = etree.fromstring(archive.read("word/_rels/document.xml.rels"))
        word_text = " ".join(xml.xpath("//w:t/text() | //m:t/text()", namespaces=NS))
        counts = {
            "source_words": len(inline_text(ast["blocks"]).split()),
            "section_headings": len(xml.xpath("//w:pPr/w:pStyle[@w:val='Heading1']", namespaces=NS)),
            "subheadings": len(xml.xpath("//w:pPr/w:pStyle[@w:val='Heading2']", namespaces=NS)),
            "tables": len(xml.xpath("//w:tbl", namespaces=NS)),
            "table_rows": len(xml.xpath("//w:tr", namespaces=NS)),
            "diagrams": len(xml.xpath("//w:drawing", namespaces=NS)),
            "embedded_svg": len(xml.xpath("//asvg:svgBlip", namespaces=NS)),
            "native_equations": len(xml.xpath("//m:oMath", namespaces=NS)),
            "external_links": len(rels.xpath("rel:Relationship[@TargetMode='External']", namespaces=NS)),
            "contents_links": len(xml.xpath("//w:hyperlink[starts-with(@w:anchor, 'report_section_')]", namespaces=NS)),
        }
        expected_headers = list(nodes(ast, "Header"))
        assert counts["section_headings"] == sum(h["c"][0] == 2 for h in expected_headers)
        assert counts["subheadings"] == sum(h["c"][0] == 3 for h in expected_headers)
        assert counts["tables"] == len(list(nodes(ast, "Table")))
        assert counts["diagrams"] == counts["embedded_svg"] == len(images)
        assert counts["native_equations"] == len(list(nodes(ast, "Math")))
        assert counts["contents_links"] == counts["section_headings"]
        assert len(re.findall(r"Figure \d+\.", word_text)) == len(images)
        assert not re.search(r"\\[\[\]()]|\\(?:mathrm|theta|Delta|frac|widetilde)\b", word_text), "Unconverted LaTeX"
        # Every ordinary source paragraph and table-cell paragraph must remain.
        # Native equation objects use a different text serialization, so their
        # non-math text is checked as ordered inline fragments instead.
        joined = re.sub(r"\s+", "", word_text)
        for kind in ("Para", "Plain", "Header"):
            for node in nodes(ast, kind):
                if list(nodes(node, "Image")) or list(nodes(node, "Math")):
                    continue
                expected = re.sub(r"\s+", "", inline_text(node))
                assert not expected or expected in joined, f"Missing source content: {expected[:120]}"
        for relationship in rels:
            if relationship.get("Type", "").endswith("/image"):
                member = "word/" + relationship.get("Target")
                assert member in archive.namelist(), f"Missing image {member}"
        for index, original in enumerate(images, 1):
            assert archive.read(f"word/media/report-vector-{index:02}.svg") == original.read_bytes(), f"Stale SVG: {original.name}"
        for name in archive.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                etree.fromstring(archive.read(name))
    counts["bytes"] = path.stat().st_size
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = (args.output or source.with_suffix(".docx")).resolve()
    for tool in ("pandoc", "rsvg-convert"):
        if not shutil.which(tool):
            parser.error(f"{tool} is required on PATH")
    ast = json.loads(command("pandoc", "--from=markdown+tex_math_single_backslash-implicit_figures-smart", "--to=json", str(source)).stdout)
    body = copy.deepcopy(ast)
    first = body["blocks"][0]
    assert first["t"] == "Header" and first["c"][0] == 1, "Expected one report title"
    title = inline_text(first)
    body["blocks"][0] = {"t": "Para", "c": first["c"][-1]}
    for header in nodes(body, "Header"):
        header["c"][0] -= 1
    images = []
    with tempfile.TemporaryDirectory(prefix="gait-progress-docx-") as temporary:
        work = Path(temporary)
        reference, prepared = work / "reference.docx", work / "report.json"
        build_reference(reference)
        for index, image in enumerate(nodes(body, "Image"), 1):
            original = (source.parent / image["c"][-1][0]).resolve()
            assert original.suffix.lower() == ".svg", f"Expected SVG diagram: {original}"
            images.append(original)
            png = work / f"report-figure-{index:02}.png"
            command("rsvg-convert", "--width=3300", "--keep-aspect-ratio", "--background-color=white", "--output", str(png), str(original))
            image["c"][-1][0] = str(png)
            image["c"][0][2] = [["width", f"{PAGE_WIDTH}in"]]
        prepared.write_text(json.dumps(body))
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = work / "report.docx"
        command("pandoc", "--from=json", "--to=docx", f"--reference-doc={reference}", f"--resource-path={source.parent}", "--output", str(pending), str(prepared))
        style_document(pending, title)
        embed_vectors(pending, images)
        result = validate(pending, ast, images)
        shutil.copyfile(pending, output)
    result["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["graphics_sha256"] = hashlib.sha256(b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in images)).hexdigest()
    result["output"] = str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
