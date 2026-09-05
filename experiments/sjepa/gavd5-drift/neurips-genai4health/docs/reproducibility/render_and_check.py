"""Render every manuscript page and inspect metadata/text for production QA."""
from pathlib import Path
import json
import re
import pymupdf
from PIL import Image, ImageOps, ImageDraw

DOCS = Path(__file__).resolve().parents[1]
OUT = DOCS / "review/pdf_qa"
OUT.mkdir(parents=True, exist_ok=True)
results = []
for name in ("genai4health_paper_draft", "genai4health_extended_abstract"):
    pdf_path = DOCS / f"{name}.pdf"
    if not pdf_path.exists():
        continue
    doc = pymupdf.open(pdf_path)
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    page_texts = []
    thumbs = []
    refs_page = None
    outside = []
    for i, page in enumerate(doc):
        text = page.get_text()
        page_texts.append(text)
        if refs_page is None and re.search(r"(?m)^References\s*$", text):
            refs_page = i+1
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.7, 1.7), alpha=False)
        pix.save(folder / f"page_{i+1:02d}.png")
        thumb = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        thumb.thumbnail((306, 420))
        canvas = Image.new("RGB", (326, 450), "#dde3e8")
        canvas.paste(thumb, ((326-thumb.width)//2, 20))
        ImageDraw.Draw(canvas).text((12, 432), f"Page {i+1}", fill="black")
        thumbs.append(canvas)
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    x0,y0,x1,y1 = span["bbox"]
                    if x0 < 0 or y0 < 0 or x1 > page.rect.width or y1 > page.rect.height:
                        outside.append({"page":i+1,"text":span["text"]})
    all_text = "\n".join(page_texts)
    (folder / "extracted_text.txt").write_text(all_text, encoding="utf-8")
    cols = 3
    rows = (len(thumbs)+cols-1)//cols
    montage = Image.new("RGB", (cols*326, rows*450), "white")
    for i, thumb in enumerate(thumbs): montage.paste(thumb, ((i%cols)*326, (i//cols)*450))
    montage.save(folder / "contact_sheet.png")
    record = {"pdf": pdf_path.name, "pages":len(doc), "references_start_page":refs_page,
              "main_text_pages":refs_page-1 if refs_page else None,
              "metadata":doc.metadata, "outside_page_text":outside,
              "anonymity_string_hits":[s for s in ("alexm", "brilliantbeaver", "C:\\Users", "ALEXPOSE_ROOT") if s.lower() in all_text.lower()],
              "replacement_glyphs": all_text.count("\ufffd"), "unresolved_citation_markers": all_text.count("[?]")}
    results.append(record)
assert all(not r["outside_page_text"] and not r["anonymity_string_hits"] and not r["replacement_glyphs"] and not r["unresolved_citation_markers"] for r in results)
(OUT / "qa_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
