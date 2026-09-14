# Writing and checking documentation

Use the [tutorial reading guide](../tutorials/reading-guide.md) to choose a study. This page describes how to keep its equations, figures and navigation readable.

## Equations

In Markdown and notebook Markdown cells, use `$...$` for inline mathematics and `$$` on separate lines for a displayed equation. Leave a blank line before and after a display. Keep executable code in code fences and explanatory mathematics outside them.

Use built-in functions such as `\log`, `\min` and `\max`, and `\mathrm{softmax}` or `\mathrm{Cov}` for other function names. Use `\text{...}` for words within equations. Use these forms consistently throughout the documentation.

For a multiline equation, use an `aligned` environment. A line beginning with a plus sign can become a Markdown list; a standalone equals sign can become a heading underline. Both can leave an apparently well-delimited equation unrendered.

```latex
$$
\begin{aligned}
\mathcal{L} &= \mathcal{L}_{\mathrm{JEPA}} \\
&\quad + \beta\mathcal{L}_{\mathrm{future}}.
\end{aligned}
$$
```

In table cells, use `\mid`, `\lvert` and `\rvert` instead of a raw pipe character that Markdown could interpret as a column boundary. Define symbols near their first use and explain what an equation measures before discussing its limitations.

## Figures and file references

Resolve relative links from the document's directory, including notebook Markdown links. Use descriptive link labels that match the destination's current name. Give images useful alternative text and keep the editable figure source near its provenance instructions. SVG text must contain rendered symbols or ordinary words, rather than LaTeX commands that a browser cannot typeset.

Link only to files available in the documented environment. For an absent historical artifact, preserve its original path and explain its availability in the [evidence inventory](evidence-availability.md). A report copy, a current source notebook and an executed notebook serve different purposes; label them accordingly.

Saved execution notebooks with recorded hashes retain their bytes. The [historical notebook guide](historical-notebooks.md) provides maintained navigation for identified files; the [registry](notebook-navigation.json) binds those replacements to exact notebook hashes. This does not authorize a blanket exception for new broken links.

## Run the checks

The validator requires Python and a local `pandoc` executable. From the repository root:

```bash
.venv/bin/python scripts/workspace_management/validate_documentation.py \
  --report work/documentation-check.json \
  --preview work/documentation-math.html

.venv/bin/python -m unittest tests.infrastructure.test_documentation -v
```

It checks Markdown, notebook Markdown, TeX manuscript references and SVGs. It compares Markdown parsing with mathematical parsing, renders expressions to native MathML, and checks local links, anchors, attachments and SVG syntax. Open the generated HTML for visual inspection. This is not a full publication build or a guarantee that every Markdown viewer supports mathematics.

The default scan includes historical authored documents and saved execution notebook Markdown. It excludes runtime/cache trees such as `work/` and `outputs/`, environment files, notebook code and outputs, and exact binary replay archives. Selected files or directories can be passed explicitly. Remote URLs are listed but not fetched; missing paths written as code are reported for review because many are examples or generated outputs.

The normal check reports the original stale links in registered historical notebooks alongside their verified companion destinations. Add `--strict-archive-links` to fail on those original links as well. Unknown broken clickable links fail in both modes.

The [13 September 2026 audit record](documentation-review-2026-09-13.json) lists repairs, independent review, preservation checks and coverage limits.
