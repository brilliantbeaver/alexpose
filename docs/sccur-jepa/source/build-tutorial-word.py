"""Build the offline handbook using Pandoc, rsvg-convert, and python-docx."""
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/sccur-jepa"
LESSONS = BASE / "tutorial"
REPOSITORY = "https://github.com/brilliantbeaver/alexpose"
NAVY = "193B59"
TEAL = "00796D"
CHAPTERS = [
    ("start", "Start here", "data-units", "One recording can supply several sequences and windows."),
    ("classifier-study", "Learning from hidden joints", "masked-learner", "The visible skeleton helps the student predict the teacher’s description of a covered joint."),
    ("restart-study", "Forecasting the next step", "forecast-window", "The model receives past positions, then predicts the future positions."),
    ("forecast-study", "Checking an unusual gait score", "ranking-pairs", "Each pair compares one recording with a normal annotation and one with an abnormal annotation."),
]


def portable(match):
    label, target = match.groups()
    if target.startswith(("http", "#")):
        return match.group(0)
    filename = Path(target).stem
    if any(filename == chapter[0] for chapter in CHAPTERS) and "/" not in target:
        return f"[{label}](#{filename})"
    destination = (LESSONS / target).resolve()
    if not destination.exists():
        raise FileNotFoundError(destination)
    kind = "tree" if destination.is_dir() else "blob"
    return f"[{label}]({REPOSITORY}/{kind}/main/{quote(destination.relative_to(ROOT).as_posix())})"


def combine():
    activities = re.split(r"(?m)^## ", (LESSONS / "activity-pages.md").read_text())[1:]
    sections = []
    for idx, (slug, short_title, figure, caption) in enumerate(CHAPTERS):
        source = (LESSONS / f"{slug}.md").read_text()
        title, content = source.split("\n", 1)
        content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", portable, content).strip()
        if idx == 0:
            navigation = "  ·  ".join(f"[{name}](#{key})" for key, name, _, _ in CHAPTERS[1:])
            first, rest = content.split("\n\n",1)
            content = first + "\n\n**Explore the handbook**\n\n" + navigation + "\n\n" + rest
        # Place the diagram beside its exercise, after the introductory material.
        heading = re.search(r"(?m)^## ", content)
        split = heading.start() if heading else len(content)
        activity_title, explanation = activities[idx].split("\n",1)
        exercise = f"\n\n## {activity_title.strip()}\n\n![{caption}]({figure}.svg){{width=6.45in}}\n\n{explanation.strip()}\n\n"
        sections.append(f"{title} {{#{slug}}}\n\n"+content[:split]+exercise+content[split:])
    return "\n\n".join(sections)


def format_document(input_file, output):
    doc = Document(input_file)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin, sec.bottom_margin = Inches(.76), Inches(.70)
    sec.left_margin, sec.right_margin = Inches(.84), Inches(.84)
    sec.header_distance, sec.footer_distance = Inches(.39), Inches(.34)
    for name in ("Normal","Body Text","First Paragraph","Compact","Caption"):
        if name not in doc.styles:
            continue
        st=doc.styles[name]
        st.font.name="Calibri"
        st.font.size=Pt(11 if name != "Caption" else 9.5)
        st.font.color.rgb=RGBColor.from_string("233747")
        st.paragraph_format.line_spacing=1.10
        st.paragraph_format.space_before=Pt(0)
        st.paragraph_format.space_after=Pt(5)
        st.paragraph_format.widow_control=True
    for name,size,color in [("Heading 1",27,NAVY),("Heading 2",16,TEAL),("Heading 3",14,NAVY)]:
        st=doc.styles[name]
        st.font.name="Calibri"; st.font.size=Pt(size); st.font.bold=True
        st.font.color.rgb=RGBColor.from_string(color)
        st.paragraph_format.space_before=Pt(12); st.paragraph_format.space_after=Pt(4)
        st.paragraph_format.keep_with_next=True
    for i,p in enumerate(doc.paragraphs):
        if p.style.name=="Heading 1":
            if i: p.paragraph_format.page_break_before=True
            else:
                p.style=doc.styles["Title"]
                for r in p.runs:
                    r.font.name="Calibri"; r.font.size=Pt(30); r.font.bold=True;r.font.color.rgb=RGBColor(0,0,0)
        if p._p.xpath('.//w:drawing'):
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next=True
        if p.style.name in ("Image Caption","Caption"):
            p.paragraph_format.space_before=Pt(1);p.paragraph_format.space_after=Pt(5)
    header=sec.header.paragraphs[0]
    r=header.add_run("GAIT ANALYSIS     /     STUDENT HANDBOOK")
    r.font.name="Calibri"; r.font.size=Pt(9);r.bold=True;r.font.color.rgb=RGBColor.from_string(TEAL)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=footer.add_run("Gait Analysis   /   ");r.font.size=Pt(9);r.font.color.rgb=RGBColor.from_string(NAVY)
    field=OxmlElement("w:fldSimple");field.set(qn("w:instr"),"PAGE");footer._p.append(field)
    for tab in doc.tables:
        tab.alignment=WD_TABLE_ALIGNMENT.CENTER
        for i,row in enumerate(tab.rows):
            pr=row._tr.get_or_add_trPr()
            pr.append(OxmlElement("w:cantSplit"))
            if i==0:pr.append(OxmlElement("w:tblHeader"))
            for cell in row.cells:
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if i%2==0:
                    shading=OxmlElement("w:shd");shading.set(qn("w:fill"),NAVY if i==0 else "F1F6F6");cell._tc.get_or_add_tcPr().append(shading)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after=Pt(3);para.paragraph_format.space_before=Pt(3)
                    for run in para.runs:
                        run.font.name="Calibri";run.font.size=Pt(10)
                        if i==0:run.bold=True;run.font.color.rgb=RGBColor.from_string("FFFFFF")
    doc.core_properties.title="How to study walking with video"
    doc.core_properties.subject="A student handbook on gait classification and forecasting"
    doc.save(output)


def main():
    with tempfile.TemporaryDirectory() as folder:
        tmp=Path(folder)
        for _,_,figure,_ in CHAPTERS:
            (tmp/f"{figure}.svg").write_bytes((LESSONS/"figures"/f"{figure}.svg").read_bytes())
        src=tmp/"tutorial.md";src.write_text(combine())
        raw=tmp/"export.docx"
        subprocess.run(["pandoc",str(src),"-f","markdown-auto_identifiers+header_attributes","--resource-path",str(tmp),"-o",str(raw)],check=True,cwd=tmp)
        format_document(raw,BASE/"Gait-Video-Tutorial.docx")

if __name__=="__main__":
    main()
