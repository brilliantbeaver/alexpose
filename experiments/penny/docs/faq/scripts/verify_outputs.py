#!/usr/bin/env python3
"""Check FAQ source links, retained evidence, and matching Word content.

This checks document integrity. It does not rerun the gait experiments.
Run with the document environment: .venv/bin/python scripts/verify_outputs.py
"""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from urllib.parse import unquote, urlparse

from lxml import etree
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "docs/faq.md"
WORD = ROOT / "docs/faq.docx"
OUTPUT = ROOT / "docs/faq/reviews/final-verification.json"


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    source = SOURCE.read_text()
    tokens = MarkdownIt("commonmark").enable("table").parse(source)
    links, images, paragraphs = [], [], []
    for token in tokens:
        if token.type != "inline":
            continue
        content = []
        for child in token.children or []:
            if child.type == "link_open":
                links.append(child.attrGet("href"))
            elif child.type == "image":
                images.append(child.attrGet("src"))
            elif child.type in ("text", "code_inline"):
                content.append(child.content)
            elif child.type in ("softbreak", "hardbreak"):
                content.append(" ")
        if content:
            paragraphs.append(normalize("".join(content)))
    missing = []
    for target in links + images:
        if not urlparse(target).scheme and not target.startswith("#"):
            path = (SOURCE.parent / unquote(target.split("#", 1)[0])).resolve()
            if not path.exists():
                missing.append(target)
    assert not missing, f"Missing relative links: {missing}"

    manifest = json.loads((ROOT / "docs/faq/evidence/provenance-manifest.json").read_text())
    for item in manifest["artifacts"]:
        data = (ROOT / item["retained_copy"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["retained_copy"]

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    with zipfile.ZipFile(WORD) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
        word_paragraphs = []
        for paragraph in document.xpath("//w:p", namespaces=ns):
            content = []
            for node in paragraph.iter():
                name = etree.QName(node).localname
                if name == "t":
                    content.append(node.text or "")
                elif name in ("br", "cr", "tab"):
                    content.append(" ")
            word_paragraphs.append(normalize("".join(content)))
        missing_text = [p for p in paragraphs if p not in word_paragraphs]
        assert not missing_text, f"Markdown content absent from Word: {missing_text[:5]}"
        assert len(document.xpath("//asvg:svgBlip", namespaces=ns)) == len(images)
        vectors = [name for name in archive.namelist() if name.endswith(".svg")]
        assert len(vectors) == len(images)
        for vector, original in zip(vectors, images):
            assert archive.read(vector) == (SOURCE.parent / original).read_bytes(), vector

    questions = re.findall(r"^### (\d+)\.", source, re.M)
    assert questions == [str(n) for n in range(1, 46)]
    captions = re.findall(r"^\*Figure (\d+)\.", source, re.M)
    assert captions == [str(n) for n in range(1, 16)]
    report = {
        "status": "passed",
        "faq_questions": len(questions),
        "inline_vector_figures": len(images),
        "retained_evidence_files_verified": len(manifest["artifacts"]),
        "local_and_external_links_parsed": len(links),
        "markdown_text_blocks_matched_in_word": len(paragraphs),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "docx_sha256": hashlib.sha256(WORD.read_bytes()).hexdigest(),
        "scope": "Relative links, evidence hashes, all Markdown text blocks, original SVG bytes, question and caption numbering. No experiment rerun.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
