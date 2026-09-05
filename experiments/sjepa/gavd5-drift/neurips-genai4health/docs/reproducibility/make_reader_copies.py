"""Create Markdown reader copies from canonical LaTeX using Pandoc.

Run: uv run --no-project --with pypandoc-binary python <this script>
The paper source and bibliography remain canonical.
"""
from pathlib import Path
import re
import pypandoc

DOCS = Path(__file__).resolve().parents[1]
for stem in ("genai4health_paper_draft", "genai4health_extended_abstract"):
    source = (DOCS / f"{stem}.tex").read_text(encoding="utf-8")
    # Include the abstract explicitly; Pandoc's LaTeX reader stores it as metadata.
    abstract = re.search(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", source).group(1).strip()
    stripped = re.sub(r"\\begin\{abstract\}[\s\S]*?\\end\{abstract\}", "", source)
    for key, value in {"app:cohort":"A", "app:methods":"B", "fig:findings":"1", "fig:weighting":"2", "tab:cohort":"2"}.items():
        stripped = stripped.replace("\\ref{" + key + "}", value)
    markdown = pypandoc.convert_text(stripped, "gfm+tex_math_dollars", format="latex",
        extra_args=["--citeproc", f"--bibliography={DOCS / 'references.bib'}", "--wrap=none"])
    markdown = markdown.replace('\r\n', '\n').replace('\r', '\n')
    # Pandoc preserves relative image paths from graphicspath inconsistently.
    markdown = re.sub(r'(?<!figures/)(audited_findings|source_weighting_and_drift)\.pdf', r'figures/\1.png', markdown)
    markdown = re.sub(r'<figure[^>]*>\s*<embed src="([^"]+)"[^>]*/>\s*<figcaption>(.*?)</figcaption>\s*</figure>',
                      lambda m: f'![{m.group(2)}]({m.group(1)})', markdown, flags=re.S)
    markdown = re.sub(r'<span[^>]*>|</span>', '', markdown)
    markdown = re.sub(r'<div[^>]*>|</div>', '', markdown)
    markdown = re.sub(r'\$`(.*?)`\$', r'$\1$', markdown)
    markdown = re.sub(r'``` math\s*(.*?)\s*```', lambda m: '$$\n' + m.group(1).strip() + '\n$$', markdown, flags=re.S)
    markdown = re.sub(r'\\(?:begin|end)\{equation\}|\\label\{[^}]+\}', '', markdown)
    markdown = re.sub(r'^# ', '## ', markdown, flags=re.M)
    markdown = re.sub(r'^#### ', '### ', markdown, flags=re.M)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    markdown = re.sub(r'(?m)^\$\$\n+', '$$\n', markdown)
    markdown = re.sub(r'\n+\$\$(?=\n|$)', '\n$$', markdown)
    markdown = markdown.replace('## Cohort and fold composition', '## Appendix A. Cohort and fold composition')
    markdown = markdown.replace('## Executed method and reproducibility limits', '## Appendix B. Executed method and reproducibility limits')
    markdown = markdown.replace('## Supplementary diagnostics and excluded evidence', '## Appendix C. Supplementary diagnostics and excluded evidence')
    markdown = markdown.replace('## Claim-to-evidence record', '## Appendix D. Claim-to-evidence record')
    lead = "# Before Gait Models Inform Care: Evidence Boundaries for Predictive Health AI\n\n"
    if "extended" in stem:
        lead += "*Companion extended abstract. The workshop does not list a separate extended-abstract track.*\n\n"
    else:
        lead += "*Position-paper draft with an empirical case study. The LaTeX source is canonical.*\n\n"
    lead += "## Abstract\n\n" + abstract.replace("\\%", "%").replace("~", " ") + "\n\n"
    (DOCS / f"{stem}.md").write_text(lead + markdown, encoding="utf-8")
    print(stem + ".md")
