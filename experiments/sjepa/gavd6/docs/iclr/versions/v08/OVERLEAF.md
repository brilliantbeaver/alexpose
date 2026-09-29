# Overleaf upload bundle — manuscript v08

**Evaluating JEPA-Inspired Motion Representations through Geometry and Gait Asymmetry**

Upload `paper-v08-overleaf.zip` as a new Overleaf project. It contains the final
anonymous manuscript and appendix, the bibliography, unchanged ICLR 2027 style
and bibliography style, all eight vector PDF figures, and all seven tables. The default paper uses a concise
five-section appendix. The optional full technical reference retains all
implementation details and numerical inventories.

## Open and compile

1. On the Overleaf project dashboard, choose **New project → Existing project (.zip)**
   and select the ZIP. Upload the ZIP itself; there is no enclosing folder to
   remove.
2. Open the project settings and select **Main document: `main.tex`** and
   **Compiler: XeLaTeX**. The compiler is an Overleaf project setting; a ZIP
   cannot reliably preselect it. XeLaTeX uses the same TeX engine family as
   the Tectonic build used to verify this manuscript.
3. Recompile. Overleaf runs the required bibliography and reference passes.
   No Python execution, external figure generation, special fonts, or shell
   escape is needed to typeset the paper.

The verified reference PDF has **nine main pages and 16 total pages**:
disclosure and references occupy pages 10–11; appendices occupy pages 12–16.
Check these boundaries after compiling or editing. Overleaf's selected TeX Live
release can differ from the local build's package versions.

## Human authorship and AI disclosure

The authors have confirmed their origination of the research ideas and initial
drafts, final editing, verification of the full text, reported results and
references, and approval of the final manuscript. The disclosure in `main.tex`
records these completed human contributions alongside the AI assistance used
for later drafting, analysis, code, and review. Uploading this project does not
itself submit a paper to the conference.

## What is included

`main.tex` is an exact renamed copy of the final `paper-v08.tex`. The other
manuscript assets are also unchanged. There is one complete LaTeX document at
the project root, so sample documents and duplicate main files cannot confuse
main-document selection. All existing relative paths are preserved.

```text
main.tex
appendix.tex
references.bib
template/iclr2027_conference.sty
template/iclr2027_conference.bst
figures/                 eight vector PDF figures
tables/                  seven tables, including full technical inventories
supplement/              optional technical reference and instructions
README.md                these instructions
bundle-manifest.json     source mapping and SHA-256 checksums
```

To typeset the optional full technical reference, follow `supplement/README.md`.
It replaces the concise appendix in a separate audit copy of the project.

The figure PDFs already embed their fonts. Standard LaTeX dependencies such as
`natbib`, `fancyhdr`, and `times` are supplied by TeX Live. No customized
`latexmkrc` is needed. The bibliography source and official `.bst` are included,
so citations remain editable and are regenerated during compilation.

This is a typesetting and manuscript-editing bundle. The separately delivered
`paper-v08-source.zip` retains the editable SVGs, Matplotlib sources, and audited
data for regenerating figures and tables. Local review records, machine-specific
provenance, compiled manuscript PDFs, build caches, and experimental ledgers are
not part of the Overleaf project.

The upload ZIP was verified by clean extraction and a local Tectonic 0.17.0
build; the validation record is `qa/overleaf-package.json` in the version
directory. The project has not been uploaded to or compiled on Overleaf itself.

## Official Overleaf instructions

- [Uploading a project](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project)
- [Choosing the main document](https://docs.overleaf.com/getting-started/recompiling-your-project/the-main-document)
- [Choosing the compiler and TeX Live version](https://docs.overleaf.com/getting-started/recompiling-your-project/selecting-a-tex-live-version-and-latex-compiler)
- [How latexmkrc is used](https://docs.overleaf.com/managing-projects-and-files/the-latexmkrc-file)
