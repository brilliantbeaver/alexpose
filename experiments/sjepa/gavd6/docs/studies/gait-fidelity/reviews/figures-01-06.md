# Independent Gait Fidelity visual review: figures 01–06

Reviewed 21 September 2026. Reviewer: masking_literature agent; did not generate or edit these figures. Opened every listed PNG using view_image at its native 1000-pixel and 900-pixel versions. This is visual review, not a conclusion inferred from SVG/text-layout checks.

## Verdict

The six diagrams are uncluttered and readable, with ample whitespace and no observed text/arrow or box-edge overlaps. One rendering defect must be repaired: figure 02 has unsupported subscript glyphs. Figures 04–06 have small clarity refinements described below. Reopen changed PNGs before acceptance.

## Required change

**P1 — 02-crossed-design.png and 02-crossed-design-900.png:** In all four cells, the two subscripts after “Reference Y” render as empty square/tofu glyphs. This obscures the reference identity, which is central to the crossed design. Replace Unicode subscript numerals with “Y00”, “Y01”, “Y10”, “Y11”, or explicitly positioned SVG tspans with ordinary numerals. The arrows, labels, spacing and footer are otherwise clear at both sizes.

## Recommended refinements

**P2 — 04-mask-contract:** The “Score valid targets” header has very little horizontal padding (only a few pixels) in the last box. It does not currently cross the border but is less comfortable than neighboring boxes. Use “Score targets”, or split into two lines with enough vertical clearance. The remaining boxes and horizontal arrow routing pass.

**P2 — 05-coverage-audit:** “Both roles for every slot” could be read as promising context coverage for naturally missing joints. Use “Both roles when eligible” or explain immediately in the caption that context coverage is conditional on a detection being available, while target eligibility follows independent reference validity. Artificial stochastic masking cannot restore naturally absent inputs. Layout itself passes at both sizes.

**P3 — 03-changing-graph-masks:** Masked versus context joints are distinguished by red/teal only. Shape redundancy (small cross inside masked circles, or hollow masked circles) would improve grayscale/color-vision accessibility. This is optional, not evidence of existing clutter. All illustrated mask sets are connected under the drawn graph; L/R are explicitly anatomical labels and the diagram makes no camera-facing claim.

**P3 — 06-matched-mask-experiment:** The subtitle correctly identifies six *pretraining* arms and the footer correctly keeps direct end-to-end separate. This agrees with training.py: _phases gives direct only end_to_end; coordinate and JEPA use pretrain then readout, and _configure_phase freezes the encoder during readout. “Coordinates” can be made more self-contained as “Coordinate prediction” if it fits, but the existing figure is scientifically correct in context. No arrows overlap cells; all labels are readable. Caption should preserve the distinction between a proposed matched target-exposure experiment and the historical code's phase-specific validity rules.

## Passed checks by figure

- 01: readable typography, clear three-column grouping, footer fits; no overlaps.
- 02: layout and arrow routing pass; subscript font rendering fails as above.
- 03: three sparse skeleton graphs are readable; connected masks and anatomical labels match captions.
- 04: linear pipeline is clear; rightmost header padding should improve.
- 05: coverage audit flows left to right without crossings; qualify eligible slots.
- 06: matrix hierarchy, pretraining designation, and separate end-to-end comparator are clear and consistent with source phases.

## Reviewed artifact hashes

- `01-research-question-900.png` — SHA-256 `4e82aa73a4ab96e411e07efcd24bb0dd978ca77de4c024c2d651e3aad7d040fa`
- `01-research-question.png` — SHA-256 `753867ce3c9a064057dfbb9b683865f4c211fafb786d7775a9a4e9c31b4c3ea7`
- `02-crossed-design-900.png` — SHA-256 `945c4cec80ae633699b4f5c0ebbf993ba7988dc1262b2b9fe585e7b97e5950ec`
- `02-crossed-design.png` — SHA-256 `76af154df49ad38b06f996604af153664c999b11ecd22f0278e8325153970326`
- `03-changing-graph-masks-900.png` — SHA-256 `b8b8c2f11dbbb69bc420ffbea09e7a51bf4febf21936a9cbafdf10d9f5a9f59a`
- `03-changing-graph-masks.png` — SHA-256 `5579eb6fa4f9fe6e5a875bb0e138ca9042c04c21f49e8f41e97b72ba6b9a1e87`
- `04-mask-contract-900.png` — SHA-256 `20beb3d986e1f71b9848bdb6d3b64f09e93bde2f2b97cbe7c24ce63dfbdcd229`
- `04-mask-contract.png` — SHA-256 `7fa51132ea00de4dcdb7f1c63fdf5aa10e93a50ba5ce360ab394436d83c8d242`
- `05-coverage-audit-900.png` — SHA-256 `0261620a2a831992f47b5197aa932a5e61439c5318a38b6de632365bb624b5bd`
- `05-coverage-audit.png` — SHA-256 `3cc8e61b0bc8e9c1d227d5473989d274507398b2db4a7a84eec33267c17c8d38`
- `06-matched-mask-experiment-900.png` — SHA-256 `68583b4e77cafa1ade5ec20d8bf6dabe5f960ec28a8049fd8e149aee7ea74ed8`
- `06-matched-mask-experiment.png` — SHA-256 `2b7378922b6e368474d8dc22b415a39314c63c83507e413974867bb4ae72c0d2`

## Revision review: final visual status

Reopened all six revised native/900 PNGs using view_image. Figure 02 now uses clearly rendered Y00/Y01/Y10/Y11. Figure 04's “Score targets” and “Independent RNG” have comfortable padding. Figure 05 now says “Both roles if eligible”. All required/recommended P1/P2 visual and label fixes are resolved. No new overlap or legibility issues. Optional shape redundancy in figure 03 remains nonblocking. Final visual verdict for figures 01–06: PASS, with the optional accessibility suggestion retained.

Final hashes for changed images:

- `02-crossed-design-900.png` — SHA-256 `9be1796b80fe7ba272990b2ff87285a58a9b468af9f0868b6d3ee16d7370178e`
- `02-crossed-design.png` — SHA-256 `40558f57861eca92b46df2f2d5f9ac4425978b568f3717368c4b4bbb224f6071`
- `04-mask-contract-900.png` — SHA-256 `e6477cdf2cd960078e2f8a104e005dd01b1e1aa3e029a26fea46b45aec0bc640`
- `04-mask-contract.png` — SHA-256 `37356575d84ee99f725d003f43e99b7a76aa7c799455c737034bfea6cfe7b57e`
- `05-coverage-audit-900.png` — SHA-256 `ba9e76303f0005b6f40895a2a3f7bafca941a4ada139768eb1adea698bf885f6`
- `05-coverage-audit.png` — SHA-256 `ec7c4c375689b5bbb47e36d99935a6aa187eae2fd9cb8c97130253404c2a6451`

## Independent scientific review of masking-study.md

Reviewed the root-authored draft and checked the relevant training.py phases and query eligibility. The draft accurately describes coordinate pretraining versus direct end-to-end training, makes proposed-versus-completed work explicit, acknowledges graph masking prior art, and correctly separates the practical three-policy screen from a connectivity attribution experiment. It handles hidden-value leakage, reference validity, side-label transforms, sparse observations, natural missingness, and person-level confirmation well. The source currently gives coordinate pretraining and clean-target JEPA the same query-token eligibility, with any-valid-frame patch support and per-window reduction; this corroborates the draft's proposed comparability, while emphasizing receipts should verify every run.

Two small clarifications are recommended:

1. Define “same target exposure” precisely as matched source windows, query positions/validity, and downstream coordinate supervision, using paired draw/mask receipts within each policy. A contextual JEPA teacher sees the full valid reference during training, whereas coordinate prediction has raw coordinate targets; this is an intended target-definition difference, not identical target information.
2. The occlusion/response comparison should distinguish recoverable cases with retained side-specific evidence from deliberately ambiguous cases where two underlying motions produce the same supplied input. Deterministic correct recovery of both is impossible; evaluate uncertainty, abstention or an added observation in that stratum. The broader main proposal already covers this boundary, but a sentence here prevents the masking experiment from silently adopting an impossible endpoint.

No new substantive blocker or novelty overclaim was found. Neither the small mask-bank audit nor the design establishes a restoration benefit or statistical significance, and the draft correctly states that limitation.

## Final accessibility and scientific disposition

Opened both final figure 03 PNGs using view_image. Context joints are solid teal and masked joints are hollow red rings, with a matching legend. This provides shape/fill redundancy and remains legible at native 1000 and 900 pixels. No new clutter or line overlap. The optional P3 accessibility concern is resolved.

Re-read masking-study.md: its explicit target-exposure definition now distinguishes matched source/query/validity/readout exposure from different target contents, and requires shared batch/mask receipts. Its occlusion paragraph now separates recoverable side-specific evidence from identical-input ambiguity, with uncertainty/abstention and the limitation of a naming anchor. Both scientific recommendations are addressed. Final independent verdict: PASS, with no remaining requested changes.

Verified that figures 01, 02, 04, 05 and 06 retain their latest visually approved PNG hashes after the complete regeneration.

Final figure 03 hashes:

- `03-changing-graph-masks.png` — SHA-256 `439e24ed06330e53d045c33c88025854a14d128cb901f26d1a82cd1285c99952`
- `03-changing-graph-masks-900.png` — SHA-256 `5f30c29ad24323e39d78c4ce7a58206e17f2e83a1c52fb3c198695d2820a32f1`

Final reviewed science draft SHA-256: `f7938bc01f2cc67b2f319ab84c64ec2ac86d0bb58e716e5678968052f87d642c`.
