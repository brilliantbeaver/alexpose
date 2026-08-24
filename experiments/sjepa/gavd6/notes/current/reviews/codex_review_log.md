# Codex adversarial review log

Blocking gate after each implementation stage. Each entry records the change
reviewed, the adversarial agent's verdict, and any follow-up. The reviewer is
instructed to be skeptical and hunt for real defects (correctness bugs,
misleading output, honesty-contract violations), not to rubber-stamp.

---

## 2026-08-05 — Task #12 (STEP 0 A2): persistent three-reference-line block in nb06 cell-031

**Change reviewed.** Prepended a flag-independent "Three reference lines" block
to the TOP of `06_capstone_health_condition_classifiers.ipynb` cell-031 (before
`video_census = ...`). Before the change, cell-031 printed NO reference lines on
the default (`SJEPA_INCLUDE_AUGMENTED_NORMAL` OFF) path — only a "BLOCKED"
message. The block now prints all three baselines on EVERY run, satisfying the
PRIME DIRECTIVE that no honest number is read without:
  1. majority-class (five-class, `five_class_majority` computed live from `labels`; myopathic 47/96 ≈ 0.490),
  2. missingness-only RF shortcut floor (`all_missingness_metrics` all-96 ≈ 0.448 acc; `exp5_missingness_metrics` exp5 ≈ 0.333),
  3. 82-feature handcrafted RF ceiling (`HANDCRAFTED_EXP5_ACCURACY = 0.7619047619047619`, `HANDCRAFTED_EXP5_MACRO_F1 = 0.7283333333333333`, mirroring cell-033, video-confounded).
Also restated the five-class references beside the ON-path Lane C numbers, with
the binary normal-vs-abnormal majority explicitly labelled "(binary)" so it is
never confused with the five-class floor.

**Reviewer.** `codex:codex-rescue` (adversarial, blocking).

**Verdict: CLEAN — no defects found.** All 7 checks passed, each verified
against the actual file (not assumed):

1. Variable scope / NameError — CONFIRMED clean. `labels` (cell-012), `pd`
   (cell-001 import), `all_missingness_metrics` / `all_sequence_metrics`
   (cell-020), `exp5_missingness_metrics` / `five_class_metrics` (cell-023) are
   all defined UNCONDITIONALLY before cell-031 in both smoke and real MODE.
2. Constant drift — CONFIRMED clean. Both constants are byte-identical to the
   literals in cell-033's `comparison` frame.
3. Numerical honesty — CONFIRMED clean. `five_class_majority` is the true
   five-class majority (47/96 = 0.4896); the block distinguishes it from the
   binary `majority` computed later in the ON path.
4. Byte-identical OFF path — CONFIRMED clean. Purely additive: computes two
   constants + one local var and prints. NO file writes (no `.to_csv` /
   `.to_parquet` / `joblib.dump` / `torch.save` / `open()` added). No canonical
   artifact altered, renamed, or added.
5. Directive satisfaction — CONFIRMED clean. With the flag OFF, all three
   reference lines print before the BLOCKED message; labels are accurate
   ("shortcut floor", "video-CONFOUNDED ceiling", "NOT honest, optimistic").
6. Duplicate/contradictory ON-path output — CONFIRMED clean. Restated numbers
   reuse the same variables (no drift); the binary majority is freshly computed
   and clearly labelled "(binary)".
7. Other bugs / honesty risks — none found.

**Independent cross-check by main agent.** `ast.parse` of cell-031 succeeds;
notebook is valid JSON; edited cell has outputs cleared and `execution_count`
reset to None (correct post-edit state). Source stored as a single 15,599-char
string (normal for a large cell).

**Follow-up / action items.** None. Gate passed; proceeded to Task #13.

**Note on `.env` state.** During this work `.env` line 10 was observed set to
`SJEPA_INCLUDE_AUGMENTED_NORMAL=1` (flag ON), while the notebook's saved output
still showed the OFF path (`augmentation-normal cohort: disabled`) — i.e. stale
output from before the flag was flipped. The cell-031 edit is correct on BOTH
paths, so this does not affect the review verdict, but the notebook has not been
re-executed under the ON flag in this session.

---

## 2026-08-05 — Task #13 (DOCS): missingness-only RF definition + FSGait page fix

**Change reviewed.** Documentation-only (no code semantics changed):
1. Sharpened the "missingness-only RF" definition in 5 places to state it is the
   same balanced RandomForest fit on the pose *validity mask alone* = per-joint
   observed fraction (33 landmarks, `all_valid.mean(axis=1)`) concatenated with
   the per-frame observed fraction (64 frames, `all_valid.mean(axis=2)`) = 33 + 64
   = 97 pose-visibility features, zero gait geometry; shortcut floor 0.448 acc
   all-96 (below 0.490 majority = 47/96), 0.333 acc exp5. Files: nb06 cell-013
   markdown, `README.md`, `docs/staged_details.md`,
   `docs/staged_sjepa_gait.md`, `docs/staged_sjepa_gait.tex`.
2. Citation correction (from #11 findings): FSGait [10] page range `2248-2264`
   → `313-329` (CrossRef-authoritative) in `docs/references.bib:111` and
   `docs/staged_sjepa_gait.md:155`. The stray/wrong `arXiv:2309.01480` (an
   unrelated BadSQA speech paper) never appeared in any manuscript-facing file —
   it exists only in `notes/`, which correctly document it as wrong — so no
   citation edit was required there.

**Reviewer.** `codex:codex-rescue` (adversarial, blocking).

**Verdict: CLEAN — no defects found.** All 5 checks (A–E) passed, verified
against the actual files and the saved
`cache/artifacts/real/missingness_only_classifier_metrics.csv`:

- **A — axis→count mapping (not swapped): PASS.** Codex traced `all_valid`
  shape `[N, 64, 33]` = `[N, frames, landmarks]`; `mean(axis=1)` collapses
  frames → 33 per-joint values, `mean(axis=2)` collapses landmarks → 64
  per-frame values; 33 + 64 = 97. Prose describes axis=1 as per-joint and
  axis=2 as per-frame — correct, not swapped.
- **B — floor numbers consistent across all 5 files & match saved outputs:
  PASS.** all-96 0.448, exp5 0.333, majority 47/96 = 0.490, handcrafted
  0.762 / 0.728 all match the saved CSV and are stated identically in every
  file. No file states a different number.
- **C — FSGait page fix in both locations, no stray 2248: PASS.**
  `references.bib:111` `pages = {313--329}`; `sjepa_gait.md:155` "pp. 313–329";
  zero remaining "2248" anywhere in `docs/`.
- **D — no introduced factual error / over-claim / contradiction: PASS.**
  Missingness framed as a floor to be *exceeded* (not approached); majority
  class correctly identified as myopathic ≈ 0.49.
- **E — GAVD [11] cites only the IEEE Access DOI, no stray arXiv: PASS.**
  `references.bib:8` and `sjepa_gait.md:157` cite only
  `10.1109/ACCESS.2025.3545787`; no `arXiv:2309.01480` anywhere in `docs/`.

**Independent cross-check by main agent.** Floor numbers grounded in notebook
saved outputs (cell-020 `0.448`, cells 020/023 `0.333`, cells 031/033
`0.7619`/`0.7283`) and exact arithmetic (47/96 = 0.4896 → 0.490); none
fabricated. FSGait correction confirmed in both `references.bib` and the `.md`
bibliography; `.tex` uses the `\cite{duan2024fsgait}` key resolved from the
corrected `.bib`.

**Follow-up / action items.** None. Gate passed; Task #13 complete.

---

## 2026-08-05 — Task #14 (images/): full redesign of the nine tutorial SVGs

**Change reviewed.** All nine `images/0*.svg` diagrams (01_method_family …
09_notebook_roadmap) were redesigned for clean UI/UX on a shared design system
(`notes/current/design/diagram_design_system.md`): fixed `viewBox 0 0 1200 720`, muted
palette, ≥22px text insets, ≥56px inter-card gaps, single-bend connectors with
≥16px arrowhead gaps. No code semantics changed. The only cross-repo embed is
`README.md:7 → images/09_notebook_roadmap.svg`; `docs/staged_details.md`/`.tex`
are decoupled (they use `docs/figures/*` from `make_figures.py`), so no PDF
rebuild was in scope.

**Reviewer.** `codex:codex-rescue` (adversarial, blocking). Codex session
`019fd51e-88c1-76b3-b22c-f069ab953479`.

**Verdict: ISSUES FOUND** — A–D PASS, E FAIL (spec↔asset consistency).

- **A — Well-formedness: PASS.** All nine parse as XML, `viewBox="0 0 1200 720"`,
  no `<script>`, no rasters, no external refs.
- **B — Content accuracy vs revised protocol: PASS.** Exact 12-landmark set
  incl. heels (03:61/65), uniform/no-motion masking, normal-first progression,
  VICReg in Stage 0, group loss only in Stages 1–4 (06:82/84) — matches
  `staged_details.md:55/89/161`.
- **C — No leaky / over-claim text: PASS.** No SVG asserts an accuracy/macro-F1
  as a validated result; 08 explicitly refuses an unsupported unseen-video score
  (08:55), consistent with `staged_details.md:234`.
- **D — README integrity: PASS.** `README.md:7` resolves to the existing
  seven-step roadmap (00 Understand → 06 Probe and audit); no other .md/.tex/
  .ipynb reference to a changed image breaks.
- **E — Internal consistency: FAIL.** (1) SVGs use auxiliary hexes absent from
  the spec palette — skeleton neutrals `#8a9bab`/`#d8e0e7`, darkened accent
  strokes `#a44c26`/`#c0483a`, warm tint `#f6e2df` (03:11, 08:43); (2) roadmap
  09 flows right-to-left with a return bend, contradicting the spec's
  "no serpentine" rule (spec:50/59).

**Resolution (user chose: amend the spec to match the verified-clean assets).**
The diagrams are correct and readable; the mismatch was the *spec* being too
narrow, not the assets being wrong. Edits to `notes/current/design/diagram_design_system.md`:
  - Added an **Auxiliary tokens** subsection sanctioning the skeleton neutrals
    (fig. 03 only), hand-darkened accent-border strokes (darker shade of an
    already-sanctioned hue only), and same-family tint fills (e.g. `#f6e2df` on
    fig. 08's risk lane).
  - Added `.ws` on-dark note token (`14px #cdd8e4`) to the type ramp — a muted
    secondary line inside dark banners (fig. 06 loss sub-terms). (This hex was
    not among Codex's sampled two files; caught by a full-set hex sweep.)
  - Replaced the absolute "no serpentine" rule with a **controlled boustrophedon**
    allowance for multi-row sequences (one clean right-angle bend per row
    transition, each row strictly single-direction) — legitimising fig. 09's
    layout while still banning the tangled routing the old #09 had.

**Independent cross-check by main agent.** Swept every distinct hex across all
nine SVGs (not just Codex's two sampled files) against the amended spec: all 21
now covered, zero residual. Verified `.ws` class exists at
`06_training_step.svg:13`. No asset was repainted, so the visually-verified
diagrams (rendered + inspected at 1200×720) are unchanged.

**Follow-up / action items.** None. Gate passed after spec reconciliation;
Task #14 complete. (Re-review not required: the fix touched only the spec doc to
describe assets Codex already validated as correct under checks A–D.)

---

## 2026-08-05 — Task #15 (images/): redesign 02_sjepa_architecture.svg connectors

**Change reviewed.** Redesigned `images/02_sjepa_architecture.svg` per two explicit
user requirements: (1) move the **Predictor** box out of its own sub-row and into the
top student row, making the student path one straight horizontal spine
(Full clip → Geometry view → Drop targets → View encoder → Predictor); (2) make all
connectors as parsimonious/direct as possible, **especially the dashed EMA line**,
which previously serpentined down the full canvas and along the bottom
(`M812 200H772V612H640V584`, three bends). The grid was laid out so View encoder
(col x=624) sits directly above the EMA target encoder (col x=624), letting the EMA
line become a single straight vertical (`M704 230V…`). Predictor (top) and Gather
outputs (bottom) converge on the dark `latent CE` node via one elbow each.

**Reviewer.** `codex:codex-rescue` (fresh thread `019fd540-74c4-7b22-b399-c01680750b0e`,
job `task-msgzutbu-mksx7u`; adversarial, blocking). Codex rendered the SVG to PNG and
inspected geometry, not just source.

**Verdict: ISSUES FOUND.** Checks B (Predictor genuinely in top row — all four boxes
`y=118 h=112`) and E (semantics preserved, no leaky metric; only numerics "T x 33 x 3"
and "only 12") **PASS**. Three defects found:
  - **A FAIL** — Full clip card at `x=48` with `stroke-width:1.75` painted to x=47.125,
    0.875px inside the 48px safe margin.
  - **C FAIL** — EMA dashed line was correctly a single straight vertical (0 bends) but
    ended at y=466, not the box top y=470. (Codex also flagged the fan-out/fan-in
    connectors as 2-bend H-V-H; see resolution.)
  - **D FAIL** — every arrowhead sat only ~2px from its target box (EMA ~4px), far below
    the ≥16px rule; and the two latent-CE elbows fully overlapped on their shared landing
    segment (1002,350)→(1018,350), reading as one arrow.

**Resolution (fixes applied to the asset).**
  - Nudged Full clip to `x=49` (painted edge 48.125 ≥ 48).
  - Re-cut every `.a` connector to stop exactly 16px short of its target box edge
    (matches the sibling convention in fig 04: arrow spans source-edge → target-edge−16).
  - EMA dashed line extended to `M704 230V454` — reaches toward the EMA box with the
    same 16px arrowhead gap; still one `V`, zero bends.
  - Split the two latent-CE elbows to distinct entry rows (Predictor→y=336,
    Gather→y=364); their vertical spans are now disjoint (174–336 and 364–526, 28px gap),
    no overlap.
  - Re-rendered at 1200×720 and re-verified numerically: all 9 arrowhead gaps = 16px,
    margin cleared, EMA single-vertical intact, elbows disjoint. XML well-formed.

**User decision (fan connectors).** Codex's 2-bend objection on the fan-out
(Full clip → two rows) and fan-in (Predictor + Gather → latent CE) was put to the user.
A one-source-to-two-targets split physically requires stub+riser+entry (H-V-H); this is
the established idiom in sibling figs 04/06, and the "single-bend" rule targets tangled
routing, not symmetric fans. **User chose "Keep H-V-H fans."** No further change.

**Follow-up / action items.** None. All actionable defects (A, C-endpoint, D) fixed and
re-verified; the sole remaining Codex flag (fan bend-count) was a resolved judgment call.
Task #15 complete.
