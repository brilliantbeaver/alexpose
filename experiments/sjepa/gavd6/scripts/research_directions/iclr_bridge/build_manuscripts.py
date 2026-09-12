"""Typeset evidence-grounded ICLR drafts from their canonical Markdown sources.

Requires Pandoc and Tectonic. The official conference style is unmodified;
only its submission-status text is patched to identify an unsubmitted draft.
Compile from the repository root with --compile. Page images and build evidence
are retained in work/artifacts/iclr-bridge-2026-09-11/manuscripts/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs/studies/iclr"
WORK = ROOT / "work/artifacts/iclr-bridge-2026-09-11/manuscripts"
TECTONIC = ROOT / "work/artifacts/iclr-bridge-2026-09-11/tools/tectonic"

PREAMBLE = r"""\documentclass{article}
\usepackage{template/iclr2027_conference}
\usepackage{times}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,calc}
\usepackage{hyperref}
\usepackage{etoolbox}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\makeatletter
\patchcmd{\@maketitle}{Under review as a conference paper at ICLR 2027}
  {Research draft --- not submitted}{\relax}{\errmessage{Draft header patch failed}}
\patchcmd{\@maketitle}{Anonymous authors\\Paper under double-blind review}
  {Research draft\\11 September 2026; not submitted for review}
  {\relax}{\errmessage{Draft author patch failed}}
\makeatother
"""


def tex_fragment(markdown: str) -> str:
    """Use Pandoc for escaping, inline math, paragraphs and narrow tables."""
    markdown = markdown.replace("−", "-").replace("⁻", "-")
    markdown = markdown.replace("R²", r"$R^2$")
    # Preserve ordinary literal Unicode arrows in the paper with explicit math.
    markdown = markdown.replace("→", r"$\rightarrow$")
    result = subprocess.run(
        ["pandoc", "--from=markdown+tex_math_single_backslash+raw_tex", "--to=latex", "--wrap=none"],
        input=markdown, text=True, capture_output=True, check=True,
    )
    return result.stdout


def citation_map() -> dict[str, str]:
    text = (DOCS / "references.bib").read_text()
    records = re.split(r"(?=@(?:inproceedings|misc)\{)", text)
    pairs = {}
    for record in records:
        key = re.match(r"@\w+\{([^,]+),", record)
        url = re.search(r"url\s*=\s*\{([^}]+)\}", record)
        if key and url:
            pairs[url[1]] = key[1]
    return pairs


def scholarly_links(markdown: str) -> str:
    pairs = citation_map()

    def convert(match):
        label, target = match.group(1), match.group(2)
        if target in pairs:
            return label + r" \citep{" + pairs[target] + "}"
        return match.group(0)

    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", convert, markdown)


def figures(markdown: str) -> str:
    pattern = re.compile(r"!\[([^\]]*)\]\((figures/[^)]+)\.svg\)\s*\n\s*\*Figure (\d+)\. (.*?)\*", re.S)

    def convert(match):
        caption = tex_fragment(match.group(4)).strip()
        return ("\n\\begin{figure}[tbp]\n\\centering\n"
                f"\\includegraphics[width=\\linewidth]{{{match.group(2)}.pdf}}\n"
                f"\\caption{{{caption}}}\n\\label{{fig:{match.group(3)}}}\n\\end{{figure}}\n")

    return pattern.sub(convert, markdown)


def clean_headings(markdown: str) -> str:
    return re.sub(r"^(#{2,3}) \d+(?:\.\d+)*\.? ", r"\1 ", markdown, flags=re.M)


def build_paper() -> Path:
    source = (DOCS / "04_paper.md").read_text()
    title = source.splitlines()[0][2:]
    abstract, body = source.split("## Abstract\n", 1)[1].split("## 1. Introduction", 1)
    body = "## 1. Introduction" + body
    body, statements = body.split("## Reproducibility and evidence statement", 1)
    statements = "## Reproducibility and evidence statement" + statements
    # Internal file links belong in the reproducibility bundle, not anonymous
    # conference citations. The draft states their role without broken PDF links.
    statements = statements.replace("[tutorial](01_critique_and_research_tutorial.md)", "tutorial")
    statements = statements.replace("[panel protocol](02_cached_panel_protocol.md)", "panel protocol")
    statements += "\nThis draft still requires complete human author verification before submission.\n"
    body = figures(scholarly_links(clean_headings(body)))
    # Source headings start at level two because level one is the document title.
    body = re.sub(r"^(#{2,3}) ", lambda m: m[1][1:] + " ", body, flags=re.M)
    statements = tex_fragment(statements)
    statements = re.sub(r"\\subsection\{([^}]+)\}", r"\\section*{\1}", statements)
    text = (PREAMBLE + "\n\\title{" + tex_fragment(title).strip() + "}\n"
            "\\begin{document}\n\\maketitle\n\\begin{abstract}\n"
            + tex_fragment(abstract.strip()) + "\\end{abstract}\n"
            + tex_fragment(body) + "\n\\label{main-text-end}\n\\clearpage\n"
            "\\bibliographystyle{template/iclr2027_conference}\n\\bibliography{references}\n"
            + statements + "\n\\end{document}\n")
    path = DOCS / "04_paper.tex"
    path.write_text(text)
    return path


def build_abstract() -> Path:
    source = (DOCS / "05_extended_abstract.md").read_text()
    title = source.splitlines()[0][2:]
    body = source.split("\n\n", 2)[2]
    text = (PREAMBLE + "\n\\title{" + tex_fragment(title).strip() + "}\n"
            "\\begin{document}\n\\maketitle\n"
            "\\begin{center}\\textbf{Extended abstract}\\end{center}\n"
            + tex_fragment(body)
            + "\n\\section*{Draft status and AI use}\n"
            "The proposed distillation method has no completed real-data student result. "
            "AI assistance supported evidence organization, literature checking, "
            "code and manuscript drafting. Human author verification remains required "
            "before submission.\n\\end{document}\n")
    path = DOCS / "05_extended_abstract.tex"
    path.write_text(text)
    return path


def compile_and_render(path: Path) -> dict:
    import pymupdf
    from PIL import Image, ImageDraw

    dependencies = ROOT / "work/artifacts/iclr-bridge-2026-09-11/tex-deps"
    command = [str(TECTONIC), "--only-cached", "--keep-logs", "--keep-intermediates",
               "-Z", f"search-path={dependencies / 'files'}",
               "-Z", f"search-path={DOCS / 'template'}", str(path)]
    environment = dict(os.environ, TECTONIC_CACHE_DIR=str(dependencies / "tectonic-cache"))
    result = subprocess.run(command, cwd=DOCS, env=environment, capture_output=True, text=True)
    (WORK / f"{path.stem}-build.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Tectonic failed ({result.returncode}); see {WORK / (path.stem + '-build.log')}")
    document = pymupdf.open(path.with_suffix(".pdf"))
    pages = []
    thumbnails = []
    for index, page in enumerate(document):
        image_path = WORK / f"{path.stem}-page-{index+1:02d}.png"
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        pix.save(image_path)
        text = page.get_text()
        (WORK / f"{path.stem}-page-{index+1:02d}.txt").write_text(text)
        pages.append({"page": index+1, "characters": len(text), "image": str(image_path.relative_to(ROOT)),
                      "has_references_heading": bool(re.search(r"^REFERENCES$", text, re.M)),
                      "ends": text[-180:]})
        thumb = Image.open(image_path).convert("RGB")
        thumb.thumbnail((306, 396))
        card = Image.new("RGB", (326, 426), "#e9ecef")
        card.paste(thumb, ((326-thumb.width)//2, 20))
        ImageDraw.Draw(card).text((12, 408), f"Page {index+1}", fill="#202d3a")
        thumbnails.append(card)
    columns = min(3, len(thumbnails))
    rows = (len(thumbnails) + columns-1)//columns
    contact = Image.new("RGB", (columns*326, rows*426), "#ced4da")
    for index, thumbnail in enumerate(thumbnails):
        contact.paste(thumbnail, ((index%columns)*326, (index//columns)*426))
    contact_path = WORK / f"{path.stem}-contact.png"
    contact.save(contact_path)
    return {"source": str(path.relative_to(ROOT)), "pages": len(document), "page_inventory": pages,
            "contact": str(contact_path.relative_to(ROOT)), "compiler_command": command,
            "official_style_unmodified": True, "draft_status_patched": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if not shutil.which("pandoc"):
        raise RuntimeError("Pandoc is required")
    paths = [build_paper(), build_abstract()]
    records = [compile_and_render(path) for path in paths] if args.compile else []
    (WORK / "build-manifest.json").write_text(json.dumps({"documents": records}, indent=2) + "\n")
    print(json.dumps({"sources": [str(p.relative_to(ROOT)) for p in paths],
                      "compiled": args.compile, "documents": records}, indent=2))


if __name__ == "__main__":
    main()
