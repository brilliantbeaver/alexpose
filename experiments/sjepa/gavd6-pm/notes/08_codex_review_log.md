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
(`notes/09_diagram_design_system.md`): fixed `viewBox 0 0 1200 720`, muted
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
narrow, not the assets being wrong. Edits to `notes/09_diagram_design_system.md`:
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

## 2026-08-19, Task #16 (notes/): symmetry-number correction pass on Idea 05 and Idea 9

**Scope.** `notes/` only. Nothing under `docs/`, `slides/`, or any `.ipynb` was touched.
Single source of truth for every number: `.work_scratch/CANONICAL_SYMMETRY_FACTS.md`,
whose values were read directly from the authoritative artifact bundles under
`gavd6/work/artifacts/real`. No number in this pass was taken from existing prose.

**Primary defect found and fixed.** `notes/ideas-claude/09-reflection-equivariant-symmetry-axis/IMPLEMENTATION.md`
presented Idea 05's measured lanes as A = -0.187, C = +0.147, B = 1.000, D = -0.014 with a
mirror slope of about -0.343. Those values come from a SUPERSEDED `d0acc262` checkpoint
bundle. The authoritative `ea59fea0` values are A = -0.602, C = -0.156, B = 1.000,
D = -0.131, mirror slope -0.741. Corrected there and at every other occurrence of the
stale trio in `notes/`, including `09-.../README.md`, `09-.../METHODOLOGY.md`,
`09-.../images/fig2.svg`, `_build_nb_09a.py`, and `_build_nb_09c.py`. Note that the sign
of C flipped as well as its magnitude, which moves the Idea 9 binding bar `max(D, C)` from
a positive +0.147 to a negative -0.156; any inference that relied on a positive bar is
therefore void.

**Known decoy left in place.** `gavd6-pm/work/artifacts/real/idea5_signed_laterality_result.json`
still carries the stale values. `resolve_artifacts()` never selects it because the
configured root wins. The file was deliberately NOT deleted and is quoted nowhere.

**Conclusions made explicit.** Added a side-by-side treatment of the three verdicts and
what each one licenses (IMPLEMENTATION.md section 9a), because they were being collapsed
into "it did not work": Idea 05 returned an INFORMATIVE NULL, a valid measurement whose
answer was no; Idea 9 Arm 1 returned ARTIFACT (side-agnostic nuisance control fired), a
WITHDRAWAL of the claim rather than an answer to it, and a weaker state than a null; Idea 9
Arm 2 returned NO CREDIT, a real and large effect on 18 of 18 sources whose preregistered
guardrail failed and supplied a competing explanation. Also documented that each arm closed
a specific escape route from the one before, that the binding constraint is the COHORT
(7.5 percent between-source variance against a preregistered 30 percent) rather than the
model or the readout, and that in all three the informative element was a CONTROL rather
than the treatment.

**Superseded claims registered** (IMPLEMENTATION.md section 9b, with pointers from the
proposal pages): the stale Idea 05 numbers; the older `nb_09b` recipe that pasted an
absolute equivariance term into notebook 04, which is satisfiable by shrinking the head;
`nb_09b`'s smoke ladder numbers, which are plumbing only and are evidence about no
checkpoint; the proposal-level Idea 5 and Idea 9 gates, replaced by the hardened gates;
`nb_09c`'s future F3, which predicted an informative null for Arm 1 where the actual verdict
was ARTIFACT; the misattribution of the 7.5 percent between-source variance to Idea 05 or
`nb_05b` when it was measured in `nb_09a`; the treatment of `nb_05b` as producing an
empirical verdict when it is a futures simulator plus an external-reach scaffold with
planted fixture values and TODO loaders; and the reading of the SCORECARD composites 4.60
and 4.55 as outcomes when they are proposal-quality design scores assigned before any run.

**Builder scripts.** Per the standing rule, `_build_*.py` files were edited ONLY where they
hardcoded a wrong result number: the mirror slope `-0.343` in `_build_nb_09a.py` (two
places) and the F3 scenario provenance note in `_build_nb_09c.py`. No generated notebook
was regenerated in this pass, so the built `.ipynb` files still carry the old strings and
belong to whichever agent owns them.

**Follow-up / action items.**
  - `notes/ideas-claude/09-.../idea9_futures_bundle.json` still stores F3's scenario lane
    values (`C_r2` 0.147, `D_r2` -0.187, `binding_bar` 0.147). These are simulator INPUTS,
    not measurements, so they were left as recorded output; the surrounding prose now says
    so explicitly. Rerunning `nb_09c` would be the clean way to retire them.
  - The stale trio also appears in generated notebooks outside this folder. Not ours.

## 2026-08-19, Task #17 (notes/): refresh 10_sjepa_evolution_tutorial.md to the ea59fea0 lineage

**Scope.** `notes/10_sjepa_evolution_tutorial.md` only, plus this log entry. The file is a maintained
implementation companion to `docs/staged_evolution.md`, not an archived plan, and it was still
describing the superseded `d0acc262` run as current.

**Root cause, and it is worth remembering.** Every stale number in the file traces to ONE source: the
in-tree `gavd6-pm/work/artifacts/real` bundle, which is a complete, internally consistent, and entirely
superseded `d0acc262` run. `resolve_artifacts()` never selects it, because the configured
`GAVD_ARTIFACT_DIR` root is checked first, so every automated tool reads `ea59fea0` correctly. A human
reading those CSVs by hand gets the previous run with no warning at all. That is exactly what happened.
The file now opens with a new section 1.1 naming both in-tree decoys, explaining the resolution order,
and giving the one command that prints the resolved root.

**Corrections applied**, all verified against the resolved artifact root:
  - fingerprint `d0acc262` to `ea59fea0`, and the five-stage chain `0a14fe12 / 563b9227 / b367796d /
    e81d529a / d0acc262` to `07fb855a / 2feef215 / a7f24edf / 269c400e / ea59fea0`, with per-stage row
    counts 75, 84, 96, 143, 159 read from each checkpoint payload;
  - legacy `sjepa_normal.pt` fingerprint `fe86339a` to `dabf5dc2`, plus the fact that it survives only in
    the old in-tree `cache/artifacts/real`;
  - training health: feature standard deviation 0.414 to 0.363, mean pair cosine 0.609 to 0.660, and the
    normal-anchor chain 0.954 / 0.839 / 0.707 / 0.594 to 0.959 / 0.849 / 0.729 / 0.617;
  - geometry: silhouette 0.009 to 0.054, minimum centroid distance 0.037 to 0.026, mean centroid distance
    0.292 to 0.313, mean within-condition distance 0.120 to 0.104;
  - masking: Stage 0 mean eligible fraction 0.551 to 0.550, Stage 4 0.423 to 0.427;
  - exact-split model revision: 0.714 / 0.742 to 0.857 / 0.881, so the legacy-to-current change is
    +0.238 / +0.268 rather than +0.095 / +0.129, with balanced accuracy 0.596 to 0.891 added;
  - all-96 lane: 0.793 to 0.759 accuracy, with balanced 0.849 and macro-F1 0.803 added;
  - Lane C five class: the superseded `0.653 / 0.603 / 0.625` pair removed as current, replaced by the
    corrected two-fold mean 0.614 / 0.615 / 0.615 and pooled out-of-fold 0.616 / 0.613 / 0.610;
  - binaries: myopathic 0.944 to 1.000 and cerebral palsy 0.889 to 1.000, all four now saturated;
  - class-level: the stale claim "A2 macro-F1 0.742, stroke F1 0.333" replaced by the current per-class
    table for both lanes, where the weakest class is cerebral palsy at F1 0.545 on A1 and 0.750 on A2;
  - the class-level error listing replaced with the current error tables, 3 errors on 21 rows for A2 and
    7 on 29 for A1.

**Structural changes.** `symmetry_verdicts.csv` now sits beside `result_history.csv` in the source
hierarchy, the maintainer checklist, and the audit commands, alongside `refresh_result_history.py
--check`. Results and conclusions were rewritten evidence-first as observation, supported inference,
unsupported inference, and next valid step. A new section 13.4 integrates the three Idea 5 and Idea 9
verdicts compactly and forwards to `ideas-claude/09-.../IMPLEMENTATION.md` sections 9a and 9b rather than
duplicating them. The audit commands were also repointed from the nonexistent `gavd4-vicreg` root and the
hardcoded `cache/artifacts/real` paths to the `gavd6-pm` root and the resolved root.

**Follow-up / action items.**
  - `gavd6-pm/work/artifacts/real` remains on disk as a full `d0acc262` bundle. It is now documented as a
    decoy rather than deleted, but it will keep catching people. Consider a stamped `SUPERSEDED` marker
    file inside it.
  - `docs/figures` was already regenerated on the current lineage, verified by the presence of 0.857 and
    0.881 and the absence of 0.333, 0.742, and 0.793 in `evolution_class_f1.svg`. No figure work needed.

## 2026-08-19, Task #18 (NOTES): proposal bodies still instructed readers to use the superseded encoder

**Defect the PDFs exposed.** The four note PDFs made an internal contradiction visible. Task #16 added a
CHECKPOINT NOTE to each proposal saying `d0acc262` was superseded by `ea59fea0`, but left the proposal
BODIES naming `d0acc262` in every method step, comparator, endpoint definition, timeline gate, and figure
caption. Two of those notes even carried the workaround sentence "where this page names `d0acc262`, read
it as `ea59fea0`", which is not a correction, it is an instruction to the reader to correct the document
by hand. All of that is now fixed at the source.

**Checkpoint corrections, by file.** Every replacement below is a current-encoder, comparator,
instruction, endpoint, gate, or caption use. Nothing labelled superseded or historical was touched.
  - `ideas-claude/05-.../README.md`: 6 body replacements plus the note rewritten;
  - `ideas-claude/05-.../METHODOLOGY.md`: 5 body replacements plus the note rewritten;
  - `ideas-claude/09-.../README.md`: 18 body replacements, including the header research question, the
    primary endpoint, the pre-registered margin, the Lane D comparator row, the Day-5 gate, the Week 2 to
    3 health check, and the Fig 1 alt text and caption, plus the note rewritten;
  - `ideas-claude/09-.../METHODOLOGY.md`: 10 body replacements plus the note rewritten;
  - `ideas-claude/_shared_facts.md`, `ideas-claude/README.md`, `ideas-claude/SCORECARD.md`: the
    current-frozen-encoder statements now name `ea59fea0`;
  - the ten other proposal folders (01 to 04, 06 to 08, 10 to 12) and `_neuro_facts.md`: 105 occurrences,
    none of which was labelled superseded, so all named the current encoder and all were corrected;
  - 14 hand-authored SVGs: 27 `<text>` and `<desc>` occurrences. `d0acc262` and `ea59fea0` are both 8
    characters, so no text metric or layout changed, and every file still parses as XML.

**Stale numbers that would have become misattributed.** Renaming the fingerprint without renaming its
numbers would have created a new contradiction, so the geometry and readout figures the proposals quote
were re-read from the authoritative bundle and corrected. Old to new, verified from
`curriculum_stage_summary_augmented.csv` and `curriculum_representation_geometry.csv`:
  - final feature standard deviation 0.413745 to 0.362567 (21 sites across 9 files);
  - final mean pairwise cosine 0.609342 to 0.659870 (15 sites across 8 files);
  - minimum between-centroid distance 0.036718 to 0.025675 (3 sites in item 07);
  - normal-anchor cosine 0.954 to 0.959 after Stage 1 and 0.594 to 0.617 after Stage 4 (9 sites in items
    02 and 07);
  - `_shared_facts.md` also corrected cosine silhouette 0.008975 to 0.054398, mean within-condition
    distance 0.119521 to 0.104404, mean centroid distance 0.292119 to 0.313160, the all-96 stratified
    readout 0.793 / 0.889 / 0.821 to 0.759 / 0.849 / 0.803, the missingness-only control
    0.448 / 0.466 / 0.429 to 0.483 / 0.507 / 0.477, the video-grouped binary readout 0.849 / 0.874 to
    0.780 / 0.804 with macro-F1 0.749 and ROC-AUC 0.915, and the video-grouped five-class readout
    0.653 / 0.625 to 0.614 / 0.615 / 0.615 with pooled out-of-fold 0.616 / 0.613 / 0.610. Each superseded
    set is retained in `_shared_facts.md` in a clearly labelled superseded bullet so old prose stays
    recognizable.
  - NOT verifiable: the decoded gait scalars R-squared about 0.719 for step amplitude and about 0.154 for
    asymmetry. The current bundle ships no scalar-decodability table, so those two values are now marked
    unverified context rather than current results, and the reader is forwarded to Idea 05's measured
    informative null.

**PDF overflow fixes.** The three substantial overfull hboxes all came from unbreakable inline code:
  - Idea 5 README, 52.36pt: `nb_05b_reflection_reach_and_futures`. The full name now appears once on its
    own fenced text line and the prose uses `nb_05b`.
  - Idea 5 METHODOLOGY, 71.32pt: the linked `../../../nb_05b_reflection_reach_and_futures.ipynb`. Both
    notebook paths and both result filenames now sit in two fenced text blocks in section 9, and the link
    labels are the short names.
  - Idea 9 README, 188.49pt: one bullet carried four `new_nb_09_*` names plus
    `work/artifacts/<mode>/idea9_arm2/`. The six notebook names and the three result bundles now sit in
    one fenced text block, and the prose says `new_nb_09_00` through `new_nb_09_03`.
  All four PDFs now rebuild with zero overfull hbox and zero overfull vbox. Two pre-existing underfull
  hboxes remain, badness 1092 in Idea 5 METHODOLOGY and badness 1442 in Idea 9 METHODOLOGY. Both are in
  ordinary prose paragraphs with no code spans, they are loose-line notices rather than overflow, and
  nothing runs past the margin.

**Follow-up / action items.**
  - Six non-Markdown files under `notes/` still name `d0acc262` as the current encoder and were left
    alone per instruction: `ideas-claude/_augment_workflow.js`, `_figures_workflow.js`, `_selection.json`,
    `05-.../_build_nb_05b.py`, `09-.../_build_nb_09c.py`, and `09-.../_build_new_nb_09_series.py`. The
    three JS and JSON files are generator inputs, so regenerating from them would reintroduce the stale
    name.
  - A concurrent agent corrected `_build_nb_05a.py`, `_build_nb_09a.py`, and `_build_nb_09c.py` and
    regenerated `nb_05a`, `nb_09a`, and `nb_09c` during this pass. Their `ea59fea0` naming agrees with
    this Markdown pass, but the overlap was not coordinated.

## 2026-08-19, Task #19 (NOTES): proposals renamed the checkpoint but kept its old architecture

**The second-order drift.** Task #18 renamed the current encoder from `d0acc262` to `ea59fea0` in every
proposal body, but the proposals also DESCRIBE that encoder, and those descriptions were not renamed with
it. Twenty-three proposal passages still said the current architecture is a 64-wide embedding read by a
depth-2 encoder. The authoritative `ea59fea0` checkpoint, read straight out of
`sjepa_curriculum_final_augmented.pt`, carries `frames 64, joints 33, coordinate_dim 3, segment_length 4,
embed_dim 96, encoder_depth 4, predictor_depth 2, heads 4`. So the embedding is 96 wide and the encoder is
4 layers deep. This is the trap a reader hits: 64 is still correct as the FRAME count, and 2 is still
correct as the PREDICTOR depth, so a blanket search-and-replace on either number would have broken more
than it fixed. Every site had to be classified before it was touched.

**What we corrected, and why each one counted as a current-architecture claim.** The rule applied was
rule 1 of the request: a passage that claims to reuse, match, inherit, or describe the current project
architecture must read 96 and 4. Heads stay 4 and predictor depth stays 2 everywhere.
  - `ideas-claude/_shared_facts.md`: the shared architecture line said "embed_dim (default 64; depth 2)".
    It now states the checkpoint values outright, "embed_dim 96; encoder depth 4, predictor depth 2, 4
    heads, GELU, pre-norm", so the one file every proposal inherits from can no longer seed the drift.
  - Glossary and machinery paragraphs in proposals 01, 02, 03, 05, 06, 07, 08, 10, and 11: the
    "each token becomes a list of 64 numbers" style sentences now say 96.
  - Substrate and recipe statements in proposals 06, 08, 10, and 11, which are the load-bearing ones
    because they say "the exact architecture", "keep the S-JEPA shape the same", "reuse the existing",
    "at project scale", and "from the project's own configuration". All now read embed 96 and encoder
    depth 4.
  - `ideas-claude/SCORECARD.md`: the item 11 explanation described the fixed substrate as width 64 and
    depth 2 layers, and now describes width 96 and encoder depth 4.
  - `11-.../README.md`: the `SUBSTRATE = dict(...)` snippet became
    `dict(embed_dim=96, encoder_depth=4, predictor_depth=2, heads=4, ...)`, which also makes the predictor
    depth explicit so the next reader does not have to infer it.
  - `08-.../README.md` and `08-.../METHODOLOGY.md`: the block layout snippet allocated
    `free: slice(36, 64)`. The three named blocks keep their designed 12-number width and the unnamed
    leftover block now absorbs the rest of the wider embedding, `free: slice(36, 96)`.
  - `02-.../README.md` line 74 is the clearest illustration of why this needed care. One sentence contains
    both "a clip is resized to 64 frames", which is the frame count and was left alone, and "an embedding
    of dimension 64", which was the stale width and became 96.

**Figures, so regeneration cannot put the drift back.** `11-.../images/fig2.svg` printed
"embed 64 / depth 2" in all four rows of the matched-substrate audit. Those now read
"embed 96 / depth 4", and the `<desc>` accessibility text, which is not laid out, carries the fuller
"same embed 96 and encoder depth 4". `08-.../images/fig3.svg` said "It writes 64 numbers per token" and
now says 96, with the `<title>` and `<desc>` updated to match. The README caption for item 11 fig 2 was
updated in step. `ideas-claude/_figures_workflow.js`, which is the generator spec for that figure, said
"same embed 64 / depth 2" and now says 96 and 4, so regenerating the figure reproduces the corrected text
rather than the drift.

**Layout was measured, not assumed.** Arial renders digits at a single tabular advance width, so
"embed 96 / depth 4" rasterises to exactly 139 px, the same as "embed 64 / depth 2", and "It writes 96"
to 77 px, the same as before. Both figures were rendered before and after at 1200 px and compared row by
row: dimensions identical, differences confined to the edited text baselines (four bands in fig 2, one in
fig 3), and the horizontal ink extent unchanged on every changed row. The longer phrasing
"embed 96 / encoder depth 4" would have measured 201 px and left only about 14 px of clearance from the
neighbouring column, which is why the compact form is in the cell and the full wording is in the caption
and the `<desc>`.

**Builder defaults.** Four builders declared the project shape as a Python default argument:
`SkeletonEncoder(embed_dim=64, depth=2)`, `SJEPAGait(embed_dim=64, encoder_depth=2, predictor_depth=2)`,
and `SkeletonPredictor(encoder_dim=64, predictor_dim=64, depth=2)`. Every real instantiation in these
files passes an explicit config, `SJEPAGait(**checkpoint["config"])` or `**SMOKE_CONFIG` or
`**MODEL_CONFIG`, so the defaults are documentation and changing them alters no behaviour. They now read
96 and 4, with `predictor_depth=2` and `SkeletonPredictor(depth=2)` deliberately left at 2 because that
is the predictor depth and it is correct. `_build_nb_05a.py` also carried the documentary fallback
`FRAMES, SEGMENT_LENGTH, EMBED_DIM = 64, 4, 64`, used only when no encoder is on disk; it is now
`64, 4, 96`, matching the twin line in `_build_nb_09a.py`. The four builders were not executed, so no
notebook was regenerated by this pass.

**Stale figure numbers found along the way.** Three SVGs still displayed the superseded geometry, which is
the same defect class one layer down: the figure named `ea59fea0` while printing `d0acc262` numbers. All
were re-read from the authoritative bundle and all replacements preserve the digit count, so the layout
was verified unchanged by the same before-and-after raster comparison.
  - `11-.../images/fig2.svg` footer: reference std 0.413745 to 0.362567, mean cosine 0.609342 to 0.659870,
    from `curriculum_stage_summary_augmented.csv`;
  - `07-.../images/fig2.svg`, three text nodes plus the `<desc>` and a source comment: normal-anchor cosine
    drift 0.954 to 0.959 after Stage 1, and 0.594 to 0.617 after Stage 4, from the same table;
  - `07-.../images/fig4.svg` footer: missingness-only macro-F1 floor 0.429 to 0.477, from
    `missingness_only_classifier_metrics.csv`, and collapse-gate std 0.413745 to 0.362567.

**Residual classification.** After editing, every remaining `64` and every remaining depth-2 mention under
`notes/` was classified mechanically. 169 are the temporal frame count, patch arithmetic such as 64 over 4
giving 16, or a normalized 64-frame timeline. 29 are the predictor depth, correct at 2. 28 are unrelated
tallies. 9 are SVG rectangle geometry attributes. 1 is a synthetic calibration fixture that declares its
own small shape, `CAL_DIM 32` at depth 2. Zero ambiguous stale current-architecture statements remain. No
proposal in this set claimed a deliberate 64-wide or depth-2 deviation, so nothing was preserved on that
ground.

**Verification.** All 56 SVGs parse as XML, all 6 JS files pass `node --check`, all 3 JSON files parse,
all 9 Python files byte-compile, and `notes/` contains zero em dash characters. The four note PDFs were
rebuilt with the layout documented in `docs/README.md`, pandoc to LaTeX with letter paper, 0.65 inch
margins and 10 pt, then tectonic, which reproduces the previous page counts exactly: Idea 5 README 14,
Idea 5 METHODOLOGY 9, Idea 9 README 13, Idea 9 METHODOLOGY 7. Zero overfull hboxes in all four. The
corrected values were confirmed present in the rendered PDF text. No other proposal folder carries a PDF.
Nothing outside `notes/` was modified.
