"""Build anonymous NeurIPS PDFs and portable Overleaf projects from eight drafts.

The Markdown files are read-only inputs. Pandoc supplies the Markdown parser and
LaTeX writer; this script normalizes presentation without revising the findings.
Requires Pandoc, Tectonic, and pypdf. No shell escape or remote upload is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
FIGURES = DOCS / "figures"
WORKSPACE = DOCS.parents[1]
BUILD = WORKSPACE / "tmp" / "pdfs" / "physworld-submissions"
READER = "markdown+tex_math_dollars+tex_math_single_backslash+raw_tex-implicit_figures"

PREAMBLE = r"""\documentclass{article}
% Anonymous workshop submission. Do not use final or preprint for review.
\PassOptionsToPackage{numbers,sort&compress}{natbib}
\usepackage[dblblindworkshop]{neurips_2026}
\workshoptitle{Physical World AI: Geometry, Characteristics, and Multimodal Sensing}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx,booktabs,longtable,array,calc}
\usepackage{microtype,xcolor,newunicodechar}
\usepackage{url}
\usepackage[hidelinks,unicode]{hyperref}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{φ}{\ensuremath{\phi}}
\newunicodechar{ε}{\ensuremath{\epsilon}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\setlength{\emergencystretch}{2em}
\graphicspath{{figures/}{../figures/}}
\hypersetup{pdftitle={%TITLE%},pdfauthor={Anonymous Author(s)},pdfcreator={LaTeX},pdfsubject={Physical World AI workshop submission}}
\title{%TITLE%}
\author{Anonymous Author(s)}
\begin{document}
\maketitle
\begin{abstract}
%ABSTRACT%
\end{abstract}
"""


def run(command: list[str], **kwargs) -> str:
    result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                            capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(node) -> str:
    if isinstance(node, list):
        return "".join(plain(x) for x in node)
    if not isinstance(node, dict):
        return ""
    kind, value = node.get("t"), node.get("c")
    if kind == "Str":
        return value
    if kind in {"Space", "SoftBreak", "LineBreak"}:
        return " "
    if kind in {"Math", "Code"}:
        return value[1]
    if kind in {"Link", "Image"}:
        return plain(value[1])
    if kind == "Header":
        return plain(value[2])
    return plain(value)


def rawblock(value: str) -> dict:
    return {"t": "RawBlock", "c": ["latex", value]}


def rawinline(value: str) -> dict:
    return {"t": "RawInline", "c": ["latex", value]}


class Builder:
    def __init__(self, pandoc: str, version: int = 7):
        self.pandoc = pandoc
        self.api = None
        self.used = {}
        self.local_links = []
        self.embedded_figures = []
        self.linked_figures = []
        self.normalizations = []
        self.require_numeric_citations = False
        # Numbered manuscripts own their bibliography. V1-V6 retain the V7
        # catalog plus legacy-only sources used by those manuscripts.
        self.catalog = {}
        self.numbered_references = {}
        self.reference_source = HERE / f"paper_v{version if version >= 7 else 7}.md"
        text = self.reference_source.read_text(encoding="utf-8")
        refs = text.split("## References\n", 1)[1].split("## Appendix A.", 1)[0]
        for paragraph in re.split(r"\n\s*\n", refs.strip()):
            number = re.match(r"^\[(\d+)\]\s+", paragraph)
            source = re.sub(r"^\[\d+\]\s+", "", paragraph).strip()
            urls = re.findall(r"\]\((https?://[^)]+)\)", source)
            if number is None or len(urls) != 1:
                raise ValueError(f"{self.reference_source.name}: references require one number and one source link per entry.")
            url = urls[0]
            if url in self.numbered_references:
                raise ValueError(f"Duplicate reference URL in {self.reference_source.name}: {url}")
            self.catalog[url] = source
            self.numbered_references[url] = int(number.group(1))
        if list(self.numbered_references.values()) != list(range(1, len(self.numbered_references) + 1)):
            raise ValueError(f"{self.reference_source.name}: reference numbers must be consecutive and unique.")
        self.catalog["https://pmc.ncbi.nlm.nih.gov/articles/PMC10388506/"] = (
            "Q. Xiong, Y. Liu, J. Mo, Y. Chen, L. Zhang, Z. Xia, C. Yi, S. Jiang, and N. Xiao. "
            "[Gait asymmetry in children with Duchenne muscular dystrophy: evaluated through "
            "kinematic synergies and muscle synergies of lower limbs]"
            "(https://pmc.ncbi.nlm.nih.gov/articles/PMC10388506/). "
            "*BioMedical Engineering OnLine*, 22:75, 2023. doi:10.1186/s12938-023-01134-7.")
        self.catalog["https://arxiv.org/abs/2301.08243"] = (
            "Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., "
            "Rabbat, M., LeCun, Y., and Ballas, N. (2023). "
            "[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture]"
            "(https://arxiv.org/abs/2301.08243). arXiv:2301.08243.")
        self.catalog["https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker"] = (
            "Google. (2026). [Pose landmark detection guide]"
            "(https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker). "
            "MediaPipe documentation; 33-landmark schema.")

    def parse(self, text: str) -> dict:
        data = json.loads(run([self.pandoc, "-f", READER, "-t", "json"], input=text))
        self.api = data["pandoc-api-version"]
        return data

    def latex(self, blocks: list[dict]) -> str:
        doc = {"pandoc-api-version": self.api, "meta": {}, "blocks": blocks}
        return run([self.pandoc, "-f", "json", "-t", "latex", "--wrap=none"],
                   input=json.dumps(doc, ensure_ascii=False)).strip()

    def inline_latex(self, inlines: list[dict]) -> str:
        return self.latex([{"t": "Plain", "c": inlines}])

    def transform(self, node, citations: bool = True):
        if isinstance(node, list):
            return [self.transform(x, citations) for x in node]
        if not isinstance(node, dict):
            return node
        node = copy.deepcopy(node)
        if node.get("t") == "Link":
            attr, label, (target, title) = node["c"]
            if target.startswith(("https://", "http://")):
                if citations:
                    if target not in self.used:
                        self.used[target] = (f"ref{len(self.used) + 1:02d}", plain(label))
                    key = self.used[target][0]
                    number = re.fullmatch(r"\[(\d+)\]", plain(label))
                    if self.require_numeric_citations and number is None:
                        raise ValueError(f"V7/V8 external citations require literal [number] labels: {target}")
                    if number:
                        expected = self.numbered_references.get(target)
                        if int(number.group(1)) != expected or int(key[3:]) != expected:
                            raise ValueError(f"Citation number/order disagrees with bibliography: {target}")
                        return rawinline(rf"\citep{{{key}}}")
                    # Retain the original descriptive citation label and give
                    # it a conventional cross-reference in historical versions.
                    return rawinline(self.inline_latex(label) + rf"~\citep{{{key}}}")
                return node
            self.local_links.append({"label": plain(label), "original_target": target})
            if target.endswith(".svg"):
                name = Path(target).stem
                self.linked_figures.append(name)
                return rawinline(self.inline_latex(label) + rf" (Figure~\ref{{fig:{name}}})")
            # Repo-local audits are not public citations. Preserve the words,
            # remove unresolvable file actions from the submission PDF.
            return {"t": "Span", "c": [["", [], []], label]}
        if node.get("t") == "Table":
            spec = node["c"][2]
            text = plain(node)
            count = len(spec)
            widths = None
            if count == 3:
                if "Array shape" in text:
                    widths = [0.18, 0.26, 0.56]
                elif "Local annotation" in text:
                    widths = [0.18, 0.17, 0.65]
                elif "Outcome that would weaken" in text:
                    widths = [0.27, 0.38, 0.35]
                elif "Evidence status" in text:
                    widths = [0.27, 0.44, 0.29]
                elif "Coordinate RMSE" in text or "Future horizon" in text:
                    widths = [0.20, 0.40, 0.40]
                elif "Notebook" in text and "Question" in text:
                    widths = [0.13, 0.40, 0.47]
                elif "95%" in text or "interval" in text:
                    widths = [0.46, 0.22, 0.32]
                else:
                    widths = [0.47, 0.29, 0.24]
            if widths:
                node["c"][2] = [[entry[0], {"t": "ColWidth", "c": width}]
                                   for entry, width in zip(spec, widths)]
        if "c" in node:
            node["c"] = self.transform(node["c"], citations)
        return node

    def table_latex(self, block: dict) -> str:
        """Keep each short source table intact using an ordinary table float."""
        tex = self.latex([block])
        if "\\endhead" not in tex or "\\endlastfoot" not in tex:
            raise ValueError("Unexpected Pandoc table format; inspect before continuing.")
        tex = tex[tex.index(r"\begin{longtable}"):]
        boundary = r"\endfirsthead" if r"\endfirsthead" in tex else r"\endhead"
        first, remainder = tex.split(boundary, 1)
        _, rows = remainder.split("\\endlastfoot", 1)
        first = re.sub(r"\\begin\{longtable\}(?:\[[^\]]*\])?", r"\\begin{tabular}", first)
        rows = rows.split(r"\end{longtable}", 1)[0] + "\\bottomrule\n\\end{tabular}"
        return "\\begin{table}[htbp]\n\\centering\n\\small\n" + first + rows + "\n\\end{table}"

    def bibliography(self) -> str:
        rows = []
        for url, (key, label) in self.used.items():
            if url in self.catalog:
                source = self.catalog[url]
            else:
                raise ValueError(f"Missing bibliographic metadata for {url} ({label}).")
            doc = self.parse(source)
            content = self.latex(self.transform(doc["blocks"], citations=False))
            rows.append(rf"\bibitem{{{key}}} " + content)
        return ("\\clearpage\n\\label{physworld:references-start}\n"
                "{\\renewcommand{\\bibfont}{\\small}\n"
                "\\begin{thebibliography}{99}\n" + "\n\n".join(rows) +
                "\n\\end{thebibliography}\n}\n")

    def build(self, version: int) -> tuple[str, dict]:
        self.require_numeric_citations = version >= 7
        source = HERE / f"paper_v{version}.md"
        text = source.read_text(encoding="utf-8")
        # Repairs are restricted to syntax in the conversion stream.
        lone = len(re.findall(r"(?m)^\$\s*$", text))
        if lone:
            text = re.sub(r"(?m)^\$\s*$", "$$", text)
            self.normalizations.append(f"Normalized {lone} lone-dollar display delimiter lines.")
        if r"(1/\sqrt2)" in text:
            text = text.replace(r"(1/\sqrt2)", r"\(1/\sqrt{2}\)")
            self.normalizations.append("Repaired the un-delimited square-root expression in Figure 3 caption.")
        if any(ord(c) < 32 and c not in "\n\r\t" for c in text):
            raise ValueError(f"Control character in {source.name}")
        doc = self.parse(text)
        blocks = doc["blocks"]
        title = blocks.pop(0)
        assert title["t"] == "Header" and title["c"][0] == 1
        title_latex = self.inline_latex(title["c"][2])
        # The dated working-copy line is not part of an anonymous submission.
        while blocks and not (blocks[0]["t"] == "Header" and plain(blocks[0]) == "Abstract"):
            blocks.pop(0)
        assert blocks and plain(blocks.pop(0)) == "Abstract"
        abstract = []
        while blocks and blocks[0]["t"] != "Header":
            abstract.append(blocks.pop(0))
        abstract_latex = self.latex(self.transform(abstract))

        main, appendix = [], []
        destination = main
        i = 0
        while i < len(blocks):
            block = blocks[i]
            if block["t"] == "Header":
                heading = plain(block)
                if heading == "References":
                    i += 1
                    while i < len(blocks) and not (blocks[i]["t"] == "Header" and
                                                  plain(blocks[i]).startswith("Appendix ")):
                        i += 1
                    continue
                if heading.startswith("Appendix "):
                    destination = appendix
                    clean = re.sub(r"^Appendix [A-Z]\.\s*", "", heading)
                else:
                    clean = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", heading)
                new_inlines = self.parse(clean)["blocks"][0]["c"]
                block = copy.deepcopy(block)
                block["c"][0] -= 1
                block["c"][2] = new_inlines
            if (block["t"] in {"Para", "Plain"} and len(block["c"]) == 1 and
                    block["c"][0]["t"] == "Image"):
                image = block["c"][0]
                name = Path(image["c"][2][0]).stem
                self.embedded_figures.append(name)
                caption = image["c"][1]
                if i + 1 < len(blocks) and plain(blocks[i + 1]).startswith("Figure "):
                    caption_block = blocks[i + 1]
                    caption = caption_block["c"]
                    if len(caption) == 1 and caption[0]["t"] == "Emph":
                        caption = caption[0]["c"]
                    caption = copy.deepcopy(caption)
                    # Strip only the figure-number prefix from the inline list.
                    prefix = 0
                    while prefix < len(caption) and not (caption[prefix]["t"] == "Str" and
                                                         re.match(r"^\d+\.$", caption[prefix]["c"])):
                        prefix += 1
                    caption = caption[prefix + 1:]
                    if caption and caption[0]["t"] == "Space":
                        caption.pop(0)
                    i += 1
                cap = self.inline_latex(self.transform(caption))
                cap = cap.replace("The right-hand intervals", "The Panel C intervals")
                destination.append(rawblock("\\begin{figure}[!tbp]\n\\centering\n" +
                    rf"\includegraphics[width=\linewidth]{{{name}_print.pdf}}" + "\n" +
                    rf"\caption{{{cap}}}\label{{fig:{name}}}" + "\n\\end{figure}"))
            elif block["t"] == "Table":
                destination.append(rawblock(self.table_latex(self.transform(block))))
            else:
                destination.append(self.transform(block))
            i += 1

        main_tex, app_tex = self.latex(main), self.latex(appendix)
        # The source links to the expanded SVG. Include that linked resource
        # as a supplementary figure so the printed cross-reference resolves.
        additional = sorted(set(self.linked_figures) - set(self.embedded_figures))
        for name in additional:
            self.embedded_figures.append(name)
            app_tex += ("\n\\clearpage\n\\section{Expanded training pipeline}\n"
                        "\\begin{figure}[htbp]\n\\centering\n" +
                        rf"\includegraphics[width=\linewidth]{{{name}_print.pdf}}" + "\n" +
                        r"\caption{Expanded version of the training pipeline linked in Figure 1, showing tensor shapes, source separation, and the distinct gradient paths for the completed training recipe and the synthetic reflection extension.}" +
                        rf"\label{{fig:{name}}}" + "\n\\end{figure}\n")
        # V7 and V8 select their two workshop appendices in Markdown. Preserve the
        # additional landmark section only in the earlier expanded revisions.
        include_landmark_schematic = version in (5, 6)
        if include_landmark_schematic:
            schema_url = "https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker"
            self.used[schema_url] = (f"ref{len(self.used) + 1:02d}", "MediaPipe landmark schema")
            schema_key = self.used[schema_url][0]
            app_tex += ("\n\\clearpage\n\\section{Anatomical selections and input shape}\n"
                        "\\begin{figure}[htbp]\n\\centering\n"
                        "\\includegraphics[width=\\linewidth]{pose_landmarks_and_laterality.pdf}\n"
                        "\\caption{Landmark-selection schematic using the published MediaPipe identities"
                        + rf"~\citep{{{schema_key}}}. "
                        "All 33 positions remain allocated in the encoder input. The 12 highlighted gait "
                        "landmarks identify historical target eligibility and VICReg pooling; the latest "
                        "motion and region samplers can select other tokens. The signed target uses five "
                        "bilateral pairs, with hips supplying the pelvis reference rather than a sixth "
                        "contrast. Coordinates are hand-drawn for explanation and represent neither "
                        "measured trajectories nor disease examples.}\n"
                        "\\label{fig:pose_landmarks_and_laterality}\n\\end{figure}\n")
        if version >= 7 and set(self.used) != set(self.numbered_references):
            raise ValueError("V7/V8 citations and reference list must cover the same sources.")
        bib_tex = self.bibliography()
        latex = (PREAMBLE.replace("%TITLE%", title_latex).replace("%ABSTRACT%", abstract_latex) +
                 main_tex + "\n\\label{physworld:main-last}\n" + bib_tex)
        if app_tex.strip():
            latex += "\\clearpage\n\\appendix\n\\label{physworld:appendix-start}\n" + app_tex
        latex += "\n\\end{document}\n"
        # Tables use the template's 9-point small style, with unchanged page
        # geometry and body font. No scaling of pages/tables to evade limits.
        latex = latex.replace(r"\begin{longtable}", "{\\small\n\\begin{longtable}")
        latex = latex.replace(r"\end{longtable}", "\\end{longtable}\n}")
        typesetting_additions = []
        if self.embedded_figures:
            typesetting_additions.append("Print-sized versions of existing figures")
        if additional:
            typesetting_additions.append("Expanded pipeline linked by the Markdown rendered in an appendix")
        if include_landmark_schematic:
            typesetting_additions.append("New schematic clarifying 33 input, 12 training, and 10 target landmarks")
        info = {"version": version, "markdown": source.name,
                "markdown_sha256": sha(source), "tex": f"paper_v{version}.tex",
                "pdf": f"paper_v{version}.pdf", "title": plain(title),
                "figures": sorted({name + "_print" for name in self.embedded_figures} |
                                  ({"pose_landmarks_and_laterality"} if include_landmark_schematic else set())),
                "typesetting_additions": typesetting_additions,
                "external_references": len(self.used),
                "citation_style": "numeric, ordered by first citation; 9-point bibliography entries",
                "bibliography_source": self.reference_source.name,
                "numbered_markdown_citations_validated": version >= 7,
                "local_links_rendered_as_text": self.local_links,
                "syntax_repairs": self.normalizations}
        return latex, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", default="pandoc")
    parser.add_argument("--tectonic", default="tectonic")
    parser.add_argument("--versions", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--tex-only", action="store_true")
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    style = HERE / "neurips_2026.sty"
    if not style.exists():
        shutil.copy2(DOCS / style.name, style)
    elif sha(style) != sha(DOCS / style.name):
        raise ValueError("Existing local style differs; refusing to overwrite it.")
    records = []
    for version in args.versions:
        builder = Builder(args.pandoc, version)
        latex, info = builder.build(version)
        destination = HERE / info["tex"]
        destination.write_text("% Generated from " + info["markdown"] +
                               "; original Markdown is unchanged.\n" + latex, encoding="utf-8")
        if not args.tex_only:
            for name in info["figures"]:
                if not (FIGURES / f"{name}.pdf").is_file():
                    raise FileNotFoundError(f"Convert SVG first: {name}.pdf")
            output = run([args.tectonic, "--keep-logs", "--keep-intermediates",
                          "--outdir", str(BUILD), str(destination)], cwd=HERE)
            (BUILD / f"paper_v{version}_build.txt").write_text(output, encoding="utf-8")
            shutil.copy2(BUILD / info["pdf"], HERE / info["pdf"])
        info["tex_sha256"] = sha(destination)
        records.append(info)
        print(f"Built v{version}: {info['tex']}", flush=True)
    (BUILD / "source_conversion_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
