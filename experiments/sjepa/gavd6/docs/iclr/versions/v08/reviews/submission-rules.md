# ICLR 2027 submission rules verified for v08

Checked 26 September 2026 against the current official pages and a fresh download of the official style ZIP. This records artifact requirements; no paper was submitted and no OpenReview registration or author eligibility was verified.

The submission main text must contain at most **nine pages**. References are excluded; unlimited appendices may follow the references, but reviewers need not read them. Use the official 2027 style. Both main text and supplementary material must be anonymous; related self-citations should use third person. A single paper-plus-appendix PDF is encouraged. These requirements support nine main pages plus disclosure, references, and a clearly marked appendix for implementation detail and complete inventories. [Official author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

An AI-use section is mandatory and excluded from the page count. The policy requires both manuscript and submission-form disclosure, with authors responsible for correctness. Here the statement should cover conceptual feedback, interpretation, manuscript drafting/editing, literature search, plotting/build code, and independent assistant review. It must not claim human verification, new experiments, or independent empirical validation that did not occur. [Official AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

The downloaded template places the AI statement after the main text and before references and says it should be at most one page. Artwork must be clean and readable, with sufficiently dark lines; numbered captions follow figures and stay with them. Captions/body must make sense in black and white. The text area is 5.5 by 9 inches with the prescribed 10-point type and 11-point spacing; Times New Roman is preferred. The user's 8–10 pt figure labels are design targets, not an explicit conference requirement. [Official style ZIP](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).

The official listed deadlines are 18 September 2026 AoE for abstracts and 25 September 2026, 23:59 AoE for papers. Do not equate a completed local package with an eligible or completed submission. [Official call for papers](https://iclr.cc/Conferences/2027/CallForPapers).

## Local verification and final acceptance checks

Fresh ZIP SHA256: `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`. It exactly matches `docs/iclr/template/iclr-2027-style-files.zip`. Each stored `.sty`, `.bst`, `.tex`, and `.bib` file also matches its entry in the fresh archive. No style-file change is needed.

The v07 build history showed that merely loading `times` under Tectonic did not guarantee the expected body font. v08 should retain the verified T1 encoding setup and inspect `pdffonts`, rather than relying on package names. After compilation, count the actual main-text pages, check where disclosure/references/appendices begin, inspect all page renders including references, verify anonymous PDF metadata, and check figure font embedding and grayscale interpretation. Ordinary float or caption reflow is preferable to changing official page dimensions or text size.

The final review will distinguish compliance with these document requirements from scientific strength and from actions that require authors' own submission records.


## Reverification for the in-place framing update

The 2026-09-26 check of the current author guidelines and fresh official ZIP is
recorded in framing-literature-and-guidelines.md. The initial-submission limit
remains nine main pages; the example permits ten during rebuttal/camera-ready.
All official style hashes remain unchanged. The revised paper is nine main pages
and 23 total, with disclosure/references before the appendices.
