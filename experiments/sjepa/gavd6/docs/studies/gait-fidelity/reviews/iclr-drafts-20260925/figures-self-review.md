# Figure construction and visual inspection — 25 September 2026

Scope: new figures in `images/iclr-draft-20260925/`, produced by `scripts/build_iclr_draft_figures.py`. This is the figure author's verification, not the independent adversarial review.

## Evidence checks

- Every numerical plot is built directly from `outputs/iclr` CSV or JSON exports. SHA-256 hashes of seven inputs and the exact plotted values are retained in `figure-provenance.json`.
- Core and response contrast intervals use the recorded crossed person/seed bootstrap. Repair uses its declared person-level Student-t interval on ViTPose inputs. They are not pooled or treated as independent replications.
- The benchmark displays the no-change response predictor at 5.81083° and explains that it supplies no restored trajectory or waveform score.
- Repair means are recomputed only from ViTPose rows. All-extractor values are not substituted for the primary population.
- The data example selects the lexicographically first participant, side camera, clear observation and original orientation. It is a real exported participant aggregate, not a raw sample. All plotted source contrasts have complete prediction coverage; the builder checks matching reference curves across methods. Raw frames and reconstruction arrays are absent from the transfer and are not invented.
- Descriptive means do not carry manufactured uncertainty. All axes start at zero except signed primary improvements and the slightly padded response-example origin.

## Visual checks and revisions

All seven PNG previews were inspected at native resolution. The plot width is 6.8 inches, with labels generally 8.5–9.5 points and short secondary annotations at 8 points. SVG files retain text; vector PDF equivalents are also saved.

The first previews exposed a full-sample title/subtitle collision, compact spacing between a legend and panel titles, and primary-effect values too close to tick labels. These were fixed by adjusting explicit text anchors and plot positions. The final previews have no detected title, legend, value or tick-label overlaps. Every build also asserts that visible text remains within the figure canvas.

Distinct markers and line styles supplement color. White backgrounds, restrained grid lines and direct curve labels keep figures legible in grayscale. Full and compact versions share values, terminology and scales. The compact figures should be displayed at their intended 6.8-inch width rather than reduced into a narrow column.

A later independent review should still assess the figures in the final HTML/PDF layout, since page-level scaling and captions can introduce new problems.
