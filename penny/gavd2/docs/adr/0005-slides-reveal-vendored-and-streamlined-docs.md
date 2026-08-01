# ADR 0005: Two reveal.js decks (vendored offline) and a streamlined docs set

**Status:** accepted

## Context

iteration 1 shipped a single hand-rolled `slides.html` and three docs (a paper, a
rendered paper, and a long 367-line tutorial). iteration 2 wants clearer slides with
better vector graphics and streamlined documentation, and it should present two
audiences: a research talk for co-authors and peers, and a teaching walkthrough for a
newcomer.

## Decision

Slides use reveal.js so we get speaker notes, transitions, keyboard navigation, and
browser PDF export. reveal.js 5.1.0 (MIT) is vendored locally into `slides/reveal/` so
the decks render fully offline with no CDN dependency, matching the repo's
self-contained ethos. Two decks live in `slides/`: `research.html` (the honest
controlled-comparison story) and `teaching.html` (the pipeline walkthrough). The
speaker-view pop-out file is vendored too; a `file://` caveat is documented because
Chrome's opaque-origin rule can block the pop-out handshake (the inline notes DOM is
always present regardless).

Docs are streamlined to a tight `paper.md`/`paper.html` plus a concise per-notebook
`pipeline.md`, dropping the long tutorial (the notebooks and the teaching deck carry the
walkthrough). An ADR set and a glossary are added as living design records.

## Consequences

- The decks open by double-clicking the HTML file, no server or install needed.
- `slides/reveal/` adds a vendored library (about 1.2 MB) to the repo; it is committed so
  offline rendering is reproducible.
- If reveal.js is upgraded, re-vendor the same file subset and re-test the speaker view.
