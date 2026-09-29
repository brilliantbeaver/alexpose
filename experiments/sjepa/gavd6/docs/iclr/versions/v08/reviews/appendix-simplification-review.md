# Proposed five-page v08 appendix

Independent methods recommendation, 2026-09-26. Read the current appendix, main-text appendix references, portable-source manifest, source/Overleaf archive builders, validator, and README. No manuscript or packaging file was edited for this review.

## Recommendation

Aim for about five appendix pages by retaining the details needed to judge the study and moving exhaustive reconstruction material into a **portable, human-readable technical supplement**. Do not compress twelve pages through smaller type or narrower margins. Remove the section-by-section `\clearpage` commands and repeated descriptions of results already shown in the main text.

Target approximately 1,700–2,000 words including concise table text and captions, four compact equation blocks, two or three small tables, and the existing all-participant figure. These are layout estimates, not verified page counts.

| Appendix component | Target | Contents |
|---|---:|---|
| A. Essential implementation | 2 pages | Context-only normalization; tensor/token contract; coordinate/base/dense objectives and support; calibration; compact architecture/optimizer settings. |
| B. Data boundaries and inference | 1 page | Split inheritance and leakage controls; reference-fixed eligibility; population weights; paired crossed bootstrap versus seed-conditional person-t; adaptive development limits. |
| C. Additional evidence and reconstruction | 1 page | Compact primary/control results, essential failure-accounting definitions, notebook completion map, and an explicit file map to complete inventories and technical details. |
| D. Participant profiles | 1 page | Existing vector figure for all 14 people, its essential caption, and the 10/14, 14/14, 1/14 descriptive counts. |

If the first layout exceeds five pages, move duplicate result prose and secondary implementation derivations first. Do not cut support rules or hide inferential limitations in the technical supplement.

## Exact content that should stay in the short appendix

### A. Implementation sufficient to understand the fitted comparison

- **Preparation:** nonoverlapping 128-frame windows at 25 Hz; nominal right-knee edits before physical mirroring; cameras fixed across paired geometries; privileged rendering boxes and clean projected references. Geometry/filename screening does not establish clinical gait. The main text already gives cameras, estimators, edit levels and person counts, so avoid repeating the full factorial here.
- **Normalization:** retain `C_i = O_i ∧ ¬M_i`, coordinatewise median origin, Euclidean norm of the coordinatewise 95th-minus-5th percentile span, the independent `(0,1)` fallback for fewer than two context points/nonfinite or tiny span, application of the same endpoint transform to references, and inverse transformation at prediction. Available observations are distinct from reference visibility. Inference adds no artificial mask.
- **Shape and masking:** retain `[B,128,12,2] → [B,384,96] → [B,128,12,2]`, four frames per joint token, five channels, and the meaning of `B` as endpoint rows. Explain that time/joint slots remain, masks do not delete tokens, naming corruption changes slot contents, and preprocessing changes physical values while preserving the model interface. Only graph-time masking was fitted among the mask alternatives.
- **Representation/readout:** width 96, four encoder layers, two predictor layers, four heads; online/student encoder used downstream; teacher and predictor discarded. A fresh shared initialization fits the residual readout, predicting corrections at observed positions and absolute coordinates at missing ones. Direct jointly adapts its encoder, whereas other readout arms hold encoders fixed. This distinction must survive any shortened comparison table.
- **Objectives and reductions:** retain the normalized coordinate MSE per valid coordinate, averaged equally over supported endpoints; centered teacher/student softmax cross-entropy with temperatures 0.06/0.1; and the 0.05 VICReg weight with component weights 25/25/1. Base queries are artificial masks or tokens containing missing observations, with at least one valid reference frame. The endpoint/delta auxiliary additionally requires both states to query a token and all four reference frames valid in both patches. Tokens average within pairs and pairs equally; lack of auxiliary support must not erase base supervision. Main Eq. (feature) need not be reprinted.
- **Readout losses:** keep the dense angular-change equation with the square inside the pair expectation and a within-pair support average. Describe the original `Lx + Ls + Lg`, low-scalar `Lx + 0.1 Ls + Lg`, and dense alternatives by referring to the main definition. Geometry support is reference-fixed; predictions cannot remove eligible frames. The original comparison adds both scalar and geometry terms.
- **Calibration:** keep the two distinct coefficient rules: shared `λJ = 0.1 sqrt(Gbase/max(Gdelta,Gendpoint))`, and per-encoder/seed `λd = 0.1 sqrt(Gscalar/Gdense)` over fresh readout parameters. Define gradient energy as the sum of squared gradients across 32 training-only batches, counting unused gradients as zero, with no parameter update. Feature calibration uses seed 17; the coefficient is reused across seeds. Preserve initial, before-clipping scope and the distinction between matching low-scalar versus matching the base objective. The full gradient table, all digits and bound-source hashes can move to the supplement.
- **Optimization:** a small settings table can retain 2,000 pretraining/readout versus 4,000 direct updates, batch 16, seeds 17/29/43, hierarchical person→motion→window→pair sampling, AdamW peak rate `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, decay `0.01`, norm-one clipping, 5% warmup/cosine decay, EMA 0.99→0.999, and center momentum 0.9. Exact update-index formulas belong in the technical record. Keep the initial 10% versus 0.0013% imbalance and 11,999/12,000 clipping result visible in either the main text or this appendix, with their noncausal limit.

### B. Boundaries and statistical interpretation

Use a short four-row table: **identity/variants**, **student inputs/normalization**, **fitting/calibration**, **remaining limits**. It should preserve:

1. Canonical people inherit original train/validation/test assignments; all motions, windows, views and corruptions stay with the person. Exact duplicate motion hashes cannot cross known identities/splits. No rendered-row split.
2. The student receives only coordinates, confidence, availability and timestamps. Reference validity and intervention/person metadata are excluded. Hidden values and references do not supply normalization statistics.
3. Training pairs/calibration use training people; ViTPose and 15° are held from fitting. Development references may be read for validation but never supply optimization/calibration losses. Confirmation tensors are rejected; no confirmation result exists.
4. Unknown aliases/upstream estimator-training overlap are unverified. Reused development people and later adaptive choices preclude an untouched independent confirmation claim. Held conditions do not constitute another person cohort.

Keep the actual averaging hierarchy: conditions→windows→motions→people→seeds. Explain the 4:1 nonheld/held endpoint weights versus 2:1 nonzero-response weights, including endpoint reuse/duplicated zero as the reason. Zero-intervention pairs do not enter the nonzero response outcome. Do not replace weighted rates with pooled raw counts.

Retain reference-fixed common angular support, two-pixel segment threshold, 16-frame/80% eligibility, waveform/response failure costs 180°/720°, and the separately eligible assignment diagnostic. Other secondary costs and direction-accuracy details can move to the technical supplement when their results move there.

State 2,000 crossed bootstrap draws, RNG 731, independent resampling of people/seeds, and paired methods. Retain repair's seed-average-first person-t rule with 13 degrees of freedom, conditional on three fitted seeds. Keep retrospective H0/H1 provenance, no new one-sided test, no noninferiority/equivalence margin, no multiplicity correction, and no adjustment for adaptive experiment choice.

### C. Evidence retained for assessment

- A compact three-row table may report the declared core, response and repair comparisons. Never substitute direct/coordinate for the original direct/change comparator. Distinguish crossed intervals from the repair person-t interval and do not pool the three effects.
- A compact descriptive negative-control block is more informative than another full outcome inventory: initialized and shuffled representations must remain separated by coordinate versus original-change readout. Coordinate-only point estimates for those controls are below core JEPA; collapsing objectives would hide that fact. The evidence reviewer is preparing the exact entries. Preserve encoder adaptation and objective labels.
- Keep `S(C)=U+Cf`, with `U` weighted over the whole population, and conditional success error `U/(1-f)`. State that conditioning changes effective person weights and differs from conditioning within source families before hierarchical averaging. Zero cost retains failures with zero loss. The 720° bound and the lower-bound argument for the fixed endpoint/delta outputs remain visible; all sixteen methods at all four costs can move out of the short appendix.
- Preserve the distinction between nominal edit bins and true response magnitude. Signed responses already averaged in the compact exports cannot recover per-pair absolute-magnitude bins, raw poses, or separate assignment-failure components.
- Compress the notebook map to three or four rows: 00–07 definitions/interfaces/reconstruction; A–E comparison/control designs with only recorded graph-time/control fits completed; F response and G repair. Retain the explicit difference between source study outcomes and instructional fits. Full notebook validation was interrupted by disk exhaustion; remote notebook validation was incomplete.
- End with a clear reproduction boundary: portable files rebuild paper/figures/tables and supply exact settings; full training and complete upstream audit still require absent assets, per-example predictions/checkpoints and repository exports. Do not describe the portable supplement as a full experiment rerun.

## Move without discarding

Move the following to a separately readable `technical-details.pdf` with editable TeX and the existing machine-readable files:

- Exact rotation/gating, mask-budget/truncation, naturally missing confidence, readout layer/initialization details, VICReg variance/covariance conventions and view augmentation, coordinate-delta scale correction/coefficient, gradient guards, and exact optimizer/EMA/center schedules.
- Full calibration energies/RMS table, per-seed clipping counts, source-bound hashes, and detailed calibration restrictions.
- Six exhaustive tables: all 16 variants, all observation-by-edit groups, all observation-by-estimator groups, all reliability decompositions, all fixed-cost scores, and complete repair inventory. Preserve their TeX and CSVs, not just screenshots.
- Full fit-count inventory, every secondary outcome cost, precise repair secondary comparisons and seed-level values, expanded notebook map, and full source/claim/provenance map.

The supplement should explicitly distinguish moved, unchanged results from newly computed analyses. Moving a table must not imply its omitted rows were unimportant or selected away.

## Vector-summary opportunities

1. **Keep the current all-participant PDF/SVG.** It already gives a compact, auditable summary without selecting examples or inventing reconstructed poses. Preserve all people, seed-range semantics, weights, failure cost and panel-specific axes.
2. **Optional thin tensor/index strip:** `endpoint grid → four-frame joint patches → 384 tokens → same endpoint grid`, with a separate short label that masks keep slots and naming swaps change contents. It can replace the repeated packing equations and paragraph. Do not add another full model diagram; the main method figure already serves that purpose.
3. **Optional support/weighting strip:** `conditions → windows → motions → people → seeds`, annotated with paired resampling and 14 people/three seeds. This can replace repeated prose, but cannot replace the 4:1 versus 2:1 rule or repair's different interval.

Avoid adding another failure bar chart or primary-effect forest plot merely to make the appendix visual: the main reliability figure and the proposed compact primary table already convey them. New graphics should replace prose, be vector PDF/SVG with editable text and embedded fonts, and not shrink labels to meet the page target.

## Portable source structure and packaging consequences

Suggested additions, keeping existing `figures/`, `tables/`, `evidence/` and builders as the numerical source:

```text
technical-details.tex          # standalone wrapper
technical-details.pdf          # readable exhaustive supplement
technical/
  details.tex                  # exact implementation and full inventories
  README.md                    # short-appendix → technical-section/file map
  executed-settings.json       # selected, portable executed configuration
  calibration-record.json      # exact values, source hashes, and audit scope
```

A settings/calibration extract must identify itself as a derived portable record, retain repository-relative origins and original hashes, and describe any omitted machine paths. It must not claim a redacted file has the original byte hash. Keep private machine paths and local review logs out of the portable package.

Make the technical document standalone: its references to main feature/cost equations must resolve, for example by repeating those equations unchanged with local labels. Do not require an unbundled main `.aux` file. One canonical copy of each numerical TeX table should serve both builders and the technical document.

Concrete current packaging issues:

- `scripts/build_source_archive.py:40–48` uses an explicit whitelist and will omit a new technical directory unless extended. The current archive contains audited CSVs and figure/table builders but excludes the complete upstream analysis/configuration/ledger set. Adding prose does not change that execution boundary.
- `scripts/build_overleaf_archive.py:37–38` asserts exactly six figure and seven table dependencies. Moving inventory tables changes those counts; use actual resolved dependencies. Its reference-PDF metadata also hardcodes 23 total pages. The Overleaf paper project should compile the condensed paper; include the technical document as a clearly identified separate deliverable rather than silently appending it again.
- `scripts/validate_package.py:13–16` asserts 23 total pages and the old appendix heading. Update these and the resulting metadata, while retaining the nine-page main-text assertion and all substantive checks.
- Main text currently promises “all observation-by-estimator groups and all 16 methods” in `app:strata`, and every method/cost in `app:cost`. Update these promises to the actual technical tables, while retaining short explanatory sections under those labels. Keep `app:methods`, `app:optimization`, `app:evaluation`, `app:journey`, `app:cost` and `fig:cases` valid where the main text still cites them.
- Update README, manifests and archive instructions to distinguish the five-page paper appendix, exhaustive technical supplement, portable figure/table reconstruction, and repository-only upstream audit. Verify clean unpacked builds of both documents and resolve every referenced supplemental path.

The final scientific check should ask whether a reader can still identify what was trained, which targets were privileged, what population/denominator each estimate uses, which controls were completed, and which claims remain unresolved **without opening the technical supplement**. Exact reconstruction minutiae can require that supplement; interpretation of the study should not.

## Final review of the implemented condensation

The parent implemented a five-section appendix: data/evaluation, training specification, primary questions, diagnostics/reproduction, and participant profiles. The complete prior appendix is supplied as an optional alternative in `supplement/technical-details.tex`. This differs from the proposed standalone wrapper, but is adequately documented: `supplement/README.md` instructs readers to replace the final appendix input in a copy of the paper project and compile only one appendix at a time. Both source variants resolve their local references.

**Verdict: no open material or minor methods finding remains.** The short appendix retains the information needed to interpret the study. The full supplement retains the exact reconstruction details rather than deleting them. Main Table 1 preserves the three actual primary comparators and identifies frozen versus jointly trained encoders. All eight original model/training procedures and both original readout objectives remain in the main restoration figure; the full sixteen-row inventories are explicitly located in the supplement rather than pooled into new procedures.

### Findings and resolutions

| Finding | Resolution checked |
|---|---|
| The first condensed dense-loss description said it averaged paired changes, omitting squared reference-relative errors. | Corrected to squared error between predicted and reference paired angular changes, with division by `180²` and common leg/time support. This now matches the main definition and full supplement. |
| “Half the available tokens” was too exact for the rounded/clamped masking budget. | Changed to “about half”; the exact rule remains in the unchanged technical source. |
| A main appendix range resolved as A–B in the short version but I–A in the optional full version. | Replaced the range with “and,” correctly identifying the two sections in either version. |
| `app:journey` pointed to primary questions in short Appendix C while the promised notebook route moved to D. | The label now belongs to D, which contains the notebook route and completion limits. |
| Portable instructions omitted `--from-audit` for the new summary-figure builder, whose default mode requires absent upstream exports. | The source-archive README command now includes `--from-audit`; the builder's portable path reads the bundled provenance selections. |

### Scientific boundaries verified

- The short appendix distinguishes source-person splitting from input-shape preservation; variants inherit the split, missing slots remain, and naming corruption does not change reference anatomy. Resampling and synthetic geometry changes remain explicit.
- Privileged teacher/reference targets, observed-and-unmasked normalization statistics, train-only fitting/calibration, permitted development-reference validation reads, held conditions, unknown aliases/upstream training overlap, and no locked-person confirmation remain visible.
- Reference-fixed angular eligibility, retained failures, population averaging, 2:1 versus 4:1 strata weights, paired crossed resampling, and repair's seed-conditional person-t interval remain explicit. Adaptive reuse is not converted to confirmation or accounted for by bootstrap draws.
- Core/change versus direct/change, delta/change versus endpoint/change, and delta/dense versus delta/low-scalar remain the actual primaries. Formal nulls remain retrospective, their intervals remain two-sided, and exploratory analyses remain identified in the main text. No response-preservation or noninferiority conclusion is added.
- Shared feature calibration still carries the 10% versus 0.001303714% initial-gradient asymmetry and the 11,999/12,000 clipping fact with appropriate limits. The repair coefficient still matches low scalar at initialization. Exact formulas, reductions and optimizer schedules remain in the technical source.
- Failure accounting retains whole-population successful contribution versus conditional success error and zero-cost lower-bound meaning. The fixed-prediction/fixed-weight restriction on the endpoint/delta lower-bound comparison remains explicit. Nominal-edit groups are not relabelled as actual response-magnitude bins.
- The notebook route still distinguishes completed source results from instructional work and planned masks/refiners/re-pairing. Incomplete full notebook execution and remote validation remain disclosed.

### Graphics, preservation, and placement

Inspected the new `protocol.png` and `study-questions.png` and their builder. The protocol shows 112 fitting people, 14 reused development people, fixed endpoint shape and 384 retained token slots, and the correct aggregation order. The primary-effect graphic selects the three original comparison records, preserves candidate/comparator identities and gain direction, uses crossed intervals for the first two rows and the original person-t interval for repair, and separates response from ViTPose waveform error. It does not pool effects or add observations.

`supplement/technical-details.tex` is byte-identical to the previously audited full appendix: SHA-256 `13b06b4f8faadfdc362e624b49cb56356ad7288acd3fd86333290f4972408b12`. Thus every original equation, table, numerical value, caption and limitation remains available. Source-label checks found no unresolved references in either `main + concise appendix` or `main + optional full appendix`. Both archive builders include the technical source/README and the required numerical tables. The portable source builder also includes the new graphics builder and its exact selected-input provenance.

The current PDF contains 16 pages, with the main-text end label on page 9: nine main pages, two disclosure/reference pages and five appendix pages. This check confirms pagination and source placement, not a fresh full-page visual inspection or a clean extracted-archive execution. Those packaging/render checks remain the parent's delivery verification; this review performed no training or new statistical analysis.

Final checked source bindings:

| File | SHA-256 |
|---|---|
| `paper-v08.tex` | `3bf4e63b5fd5df08db5f0a946e3f8287f8804a62d9a74a992fc8badbb9602f5b` |
| `appendix.tex` | `ba1ac5868601a44368d436a1d38cdb638b7326e4b35107543a5425908d847bd9` |
| `supplement/technical-details.tex` | `13b06b4f8faadfdc362e624b49cb56356ad7288acd3fd86333290f4972408b12` |
| `supplement/README.md` | `9b7681a0db56986f38fd24056a6cc241b352e2ec11f583955552b4e75da5c56f` |
| `scripts/build_source_archive.py` | `b825f721eaacd4f9105151a1142cd3052cf6793c50bfa2a5b4ef788214813f8c` |
| `scripts/build_overleaf_archive.py` | `d94ff0a4aabb6eddd8ce46f85f5ee3f3ba0895da0a6ac01b9c0d7adb9e7c70d8` |
| `scripts/build_summary_figures.py` | `4b4de8b7d201cb1c33f07b24afdbaea6816df10777d7ad937ff8baaf0ab046a6` |
