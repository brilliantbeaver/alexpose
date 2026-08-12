# Slides (gavd2 iteration 2)

Two self-contained reveal.js decks. reveal.js 5.1.0 is vendored locally under
`reveal/` (MIT licensed, see `reveal/LICENSE`), so the decks render fully offline with
no CDN and no build step.

- **`research.html`** - the honest controlled-comparison talk for co-authors and peers:
  the label-scarcity thesis, the method and its two load-bearing fixes, the four
  comparability fixes, the per-sequence result versus the 0.76 Random Forest, label
  efficiency, clinical axes, and limits and future work.
- **`teaching.html`** - a newcomer walkthrough of the whole pipeline, one section per
  notebook (scan, download, extract, corpus, pretrain, probe).

All figures are pulled from `../images/`. The numbers in the decks are from the full real
run of the pipeline on the exact exp5 68 sequences (chased to 68-of-68 coverage): the
per-sequence probe reads 0.49 (linear) to 0.63 (MLP), and the exp5 exact-split matched
Random Forest reads 0.62, against the 0.762 baseline.

## Opening a deck

Double-click `research.html` or `teaching.html`, or open it in any browser. Navigation:
arrow keys or space to move, `F` for fullscreen, `O` for the slide overview, `S` for the
speaker view, `Alt-click` (or `B`) to zoom. The slide number shows current over total in
the corner.

## Speaker notes and the file:// caveat

Speaker notes are embedded in each slide inside `<aside class="notes">`, so the notes
DOM is always present. Pressing `S` opens the pop-out speaker view
(`reveal/plugin/notes/speaker-view.html` is vendored for this).

One caveat when opening from a `file://` URL: some browsers (notably Chrome) treat each
`file://` document as an opaque origin, which can block the `window.open` plus
`postMessage` plus `localStorage` handshake the speaker view relies on, so the pop-out
window may not sync. Two reliable options:

- Present in the main window; the notes are still authored per slide.
- Or serve the folder over http and open the deck there, e.g. from this `slides/` folder:

  ```bash
  python3 -m http.server 8000
  # then open http://localhost:8000/research.html and press S
  ```

## Exporting to PDF

reveal.js supports print-to-PDF from Chrome: open the deck with `?print-pdf` appended to
the URL, then use the browser's Print dialog (Save as PDF, background graphics on,
margins none). For example `research.html?print-pdf`.

## Re-vendoring reveal.js

If you upgrade reveal.js, re-fetch the same file subset into `reveal/` (dist/reveal.js,
dist/reveal.css, dist/reset.css, dist/theme/white.css, plugin/notes/notes.js,
plugin/notes/speaker-view.html, plugin/highlight/highlight.js + monokai.css,
plugin/zoom/zoom.js) and re-test the speaker view.
