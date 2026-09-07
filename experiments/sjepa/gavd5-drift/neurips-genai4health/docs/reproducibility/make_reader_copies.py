"""Create Markdown reader copies from canonical LaTeX with external Pandoc.

Uses Python's standard library and the pandoc command already on PATH. It does
not require pypandoc or install software. Run from any working directory.
"""

from pathlib import Path
import re
import shutil
import subprocess


DOCS = Path(__file__).resolve().parents[1]
STEMS = ("genai4health_paper_draft", "genai4health_extended_abstract")
REFERENCES = {
    "fig:weighting": "1", "tab:classification": "1", "tab:record": "2",
    "app:methods": "A", "app:weighting": "B", "tab:weighting_details": "3",
}


def pandoc(text: str, output_format: str, *, citations: bool = False,
           input_format: str = "latex") -> str:
    command = ["pandoc", f"--from={input_format}", f"--to={output_format}", "--wrap=none"]
    if citations:
        command += ["--citeproc", f"--bibliography={DOCS / 'references.bib'}"]
    result = subprocess.run(command, input=text, text=True, capture_output=True,
                            cwd=DOCS, check=True)
    if result.stderr.strip():
        print(result.stderr.strip())
    return result.stdout.replace("\r\n", "\n").replace("\r", "\n")


def braced_argument(source: str, command: str) -> str:
    start = source.index(command + "{") + len(command) + 1
    depth, cursor = 1, start
    while cursor < len(source):
        character = source[cursor]
        if character == "\\":
            cursor += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start:cursor]
        cursor += 1
    raise ValueError(f"Unclosed argument for {command}")


def check_table_roundtrip(source: str, markdown: str) -> int:
    """Compare every complete cell, including numbers, against canonical LaTeX."""
    original_tables = []
    for block in re.findall(r"\\begin\{tabular\}\{[^\n]*\}\n(.*?)\\end\{tabular\}", source, flags=re.S):
        rows = []
        for line in block.splitlines():
            if "&" in line:
                line = re.sub(r"\\\\\s*$", "", line)
                rows.append([" ".join(pandoc(cell.strip(), "plain").split())
                             for cell in line.split("&")])
        original_tables.append(rows)
    converted_tables = []
    for block in re.findall(r"(?:^\|[^\n]*\|\n)+", markdown, flags=re.M):
        rows = []
        for line in block.splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r"[:\- ]+", cell) for cell in cells):
                continue
            rows.append([" ".join(pandoc(cell, "plain", input_format="gfm").split())
                         for cell in cells])
        converted_tables.append(rows)
    assert len(original_tables) == len(converted_tables), "Reader copy lost a table."
    for index, (expected, observed) in enumerate(zip(original_tables, converted_tables), 1):
        assert expected == observed, ("Reader-copy table cell changed", index, expected, observed)
    if r"\label{tab:record}" in source:
        record_row = next(row for table in converted_tables for row in table
                          if row[0] == "Recording unit and weight")
        assert record_row[1].startswith("64 clips from five videos;"), "Reader copy dropped the clip count."
    return sum(len(row) for table in original_tables for row in table)


def create_reader_copy(stem: str) -> None:
    source = (DOCS / f"{stem}.tex").read_text(encoding="utf-8")
    title = " ".join(pandoc(braced_argument(source, r"\title"), "plain").split())
    abstract = re.search(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", source)
    assert abstract, f"Abstract missing in {stem}"
    # Put the abstract in the body so citations and math share one conversion pass.
    prepared = source[:abstract.start()] + "\\section*{Abstract}\n" + abstract.group(1).strip() + source[abstract.end():]
    prepared = prepared.replace(r"\maketitle", "")
    # Pandoc 3.10 drops a leading numeric token in a p-column with these
    # decorators. They affect LaTeX layout only; remove them from conversion input.
    prepared = prepared.replace(r">{\raggedright\arraybackslash}", "")
    for key in re.findall(r"\\ref\{([^}]+)\}", prepared):
        assert key in REFERENCES, f"Add an explicit reader-copy mapping for {key}"
        assert r"\label{" + key + "}" in source, f"Missing source label: {key}"
        prepared = prepared.replace(r"\ref{" + key + "}", REFERENCES[key])

    # Add appendix letters only to the derived reader copy.
    if r"\appendix" in prepared:
        before, appendix = prepared.split(r"\appendix", 1)
        appendix_index = 0

        def appendix_heading(match: re.Match) -> str:
            nonlocal appendix_index
            letter = chr(ord("A") + appendix_index)
            appendix_index += 1
            return "\\section{Appendix " + letter + ". " + match.group(1) + "}"

        appendix = re.sub(r"\\section\{([^}]+)\}", appendix_heading, appendix)
        prepared = before + appendix

    markdown = pandoc(prepared, "gfm+tex_math_dollars", citations=True)
    markdown = re.sub(r'(?<!figures/)weighting_comparison\.pdf',
                      'figures/weighting_comparison.svg', markdown)
    markdown = markdown.replace('figures/weighting_comparison.pdf',
                                'figures/weighting_comparison.svg')
    markdown = re.sub(
        r'<figure[^>]*>\s*<embed src="([^"]+)"[^>]*/>\s*<figcaption>(.*?)</figcaption>\s*</figure>',
        lambda match: ('![Clip-weighted and video-weighted averages.](' + match.group(1) +
                       ')\n\n**Figure 1.** ' + match.group(2).strip()), markdown, flags=re.S,
    )
    markdown = re.sub(r'<div id="refs"[^>]*>', '# References\n\n', markdown)
    markdown = re.sub(r'<span[^>]*>|</span>|<div[^>]*>|</div>', '', markdown)
    markdown = re.sub(r'(?m)^(.+?) \{#(tab:[^}]+)\}$',
                      lambda match: '**Table ' + REFERENCES[match.group(2)] + '.** ' + match.group(1), markdown)
    markdown = re.sub(r'\$' + chr(96) + r'(.*?)' + chr(96) + r'\$', r'$\1$', markdown)
    fence = chr(96) * 3
    markdown = re.sub(fence + r' math\s*(.*?)\s*' + fence,
                      lambda match: '$$\n' + match.group(1).strip() + '\n$$', markdown, flags=re.S)
    markdown = re.sub(r'\\(?:begin|end)\{equation\}|\\label\{[^}]+\}', '', markdown)
    markdown = re.sub(r'^(#{1,5}) ',
                      lambda match: '#' * (3 if len(match.group(1)) >= 4 else len(match.group(1)) + 1) + ' ',
                      markdown, flags=re.M)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    markdown = re.sub(r'(?m)^\$\$\n+', '$$\n', markdown)
    markdown = re.sub(r'\n+\$\$(?=\n|$)', '\n$$', markdown)

    if "extended" in stem:
        description = "*Companion extended abstract. The workshop does not list a separate extended-abstract track; the LaTeX source is canonical.*"
    else:
        description = "*Position-paper draft with an empirical case study. The LaTeX source is canonical.*"
    result = f"# {title}\n\n{description}\n\n" + markdown.strip() + "\n"
    checked_cells = check_table_roundtrip(source, result)
    assert "\\ref{" not in result and "[?]" not in result
    assert "{#tab:" not in result and "## References" in result
    assert not any(old in result for old in ("audited_findings", "source_weighting_and_drift", "cohort_attrition"))
    if "paper_draft" in stem:
        assert "figures/weighting_comparison.svg" in result
    (DOCS / f"{stem}.md").write_text(result, encoding="utf-8")
    print(f"Created {stem}.md from canonical LaTeX; all {checked_cells} table cells preserved: {title}")


if __name__ == "__main__":
    assert shutil.which("pandoc"), "Install/provide the external pandoc command before building."
    for name in STEMS:
        create_reader_copy(name)
