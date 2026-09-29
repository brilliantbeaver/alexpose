# Figure 1: independent methods critique and replacement contract

The initial critique below reviewed the pre-redesign equation-heavy `figures/method.png`, its `schematic()` implementation, and the caption/method equations. The later user request changed the brief from an architecture diagram to an illustrated experimental workflow. The final disposition at the end of this record supersedes the intermediate architecture recommendations. No manuscript or figure source was edited by this reviewer.

## What prevents quick understanding

The current figure is mathematically compact but functions as two adjacent equation blocks. It makes the reader decode notation before seeing the experiment:

- Panel A draws the original-to-edited relationship, then gives observations and references as separate formulas. It does not show their common physical source or where the two data paths diverge.
- Panel B hides the paired windows behind the index `i`. The central distinction—matching each state's representation versus matching their paired change—appears only after reading the residual equation.
- “EMA teacher” is named, but the student-to-teacher parameter update is not drawn. The clean projected reference is not visibly marked as privileged, training-only information.
- The two auxiliary formulas look like losses applied together. Their status as alternatives added to the same base objective is relegated to one small line.
- Panel C jumps directly to an encoder plus readout. It omits the fact that the retained student encoder receives a separately fitted, reference-supervised coordinate readout.
- `H`, `c`, temperatures, stop-gradient, dimension and normalization constants consume the visual attention that should explain data flow and the comparison. Those details are already specified in the methods.

The strongest visual message would be: **two independently restored motion states, a controlled choice of feature supervision during pretraining, and one observed window required at deployment**.

## Minimum semantic contract

### 1. Controlled pairs and two data paths

Show an original complete motion window and its synthetically knee-edited counterpart. Their relationship is an edit, not progression from a past state to a future state. Each provides:

- **Clean projected 2D references**, anatomically named, available during training.
- **Estimated 2D observations**, obtained through rendering/occlusion, pose estimation, and input naming corruption.

Both paths share the physical motion and fixed camera. Within a pair, camera and observation conditions are matched. Optional **physical mirroring acts before projection**, changing both reference and rendered geometry. **Naming corruption acts only on estimated inputs**, permuting coordinates, confidence and availability while leaving reference names unchanged. A naming-swap arrow must not enter the clean-reference lane. A physical-mirror arrow must not be shown as merely swapping 2D names or forcing the signed response to reverse.

The full `Render → Pose → naming` chain may be one labelled box. The distinction between its output and a projected reference may not be collapsed into two ordinary augmentations of the same observation.

### 2. Paired pretraining, with independent state processing

For states `a` and `b`, show estimated observations entering a **student encoder + predictor**, and projected references entering a **teacher encoder**. The same network weights process each state, but each complete window is forwarded independently. Two stacked state lanes or “apply separately; shared weights” communicates this without implying concatenated pair attention.

Include a dashed **student → teacher parameter update** labelled EMA; distinguish it visually from solid data arrows. Teacher targets are detached from backpropagation. A caption can explain stop-gradient without a separate symbol.

If normalization appears, label it **per-state, observation-derived normalization shared with that state's reference**. Artificially hidden observations are excluded when computing the transform. It must not appear that clean references set student normalization statistics. This detail can be in the caption if adding transform arrows would clutter the main idea.

Do not draw a prediction arrow from state `a` alone to state `b`'s reference. The implementation does not learn that state transition.

### 3. Shared base plus one alternative auxiliary

Label the central comparison **“Separate runs: same base feature objective + one auxiliary”**, with an explicit **OR** between:

- **Endpoint: match each state's scores.** Compare the predicted and teacher scores for `a`, and separately for `b`.
- **Delta: match the change across states.** Compare the change in student-predicted scores from `a` to `b` with the corresponding teacher-score change.

A concise formula `base + λ × endpoint` versus `base + λ × delta` is sufficient. Do not draw a sum of both auxiliaries. The architecture is shared as a design, but the two training runs have separately evolving teachers.

For accuracy, label the compared quantities **“centered/scaled feature scores”** or **“channel scores (before softmax)”**, with the transformation explained in the caption/method equation. These are not categorical class labels, physical angles, or directly calibrated motion units. The base objective uses centered teacher/student softmax cross-entropy plus the regularizer; the auxiliary squared losses act on centered, temperature-scaled channel scores. An unqualified raw-feature Euclidean-regression picture would misrepresent that combination.

Common queried tokens must be the comparison unit. Full validity/support reductions, coefficients and initial-gradient imbalance can stay in the methods. The diagram must not suggest that the figure establishes matched auxiliary influence or a superior delta result.

### 4. Readout fitting and single-state deployment

Carry the **retained student encoder**, not the teacher, into a short readout-training stage. State that the coordinate readout is fitted against clean projected references. For the JEPA variants the encoder is frozen; the direct control instead learns encoder and readout together. This exception can be in the caption.

Then draw one deployment path:

`one observed 2D window → retained encoder + trained coordinate readout → restored 2D window`

No teacher, reference, partner window or predictor enters that path. The model outputs joint coordinates, not a categorical side label or a single response scalar. Corrections at observed positions and absolute predictions at missing positions can be explained in the caption. Pairwise measurement is computed from separately restored windows when evaluation compares them.

## Suggested visual organization

Use a restrained three-part flow, prioritizing the center:

1. **Create a controlled pair** — a small source panel with original/edited windows and distinct observed/reference branches.
2. **Compare feature supervision** — the largest panel, showing paired student/teacher paths, dashed EMA, shared base objective and two alternative auxiliary cards.
3. **Fit and deploy the readout** — a bottom strip distinguishing supervised readout fitting from single-window inference.

Use consistent color roles: one for observed inputs/student, a neutral or separate tone for privileged references/teacher, and the existing endpoint/delta colors for the alternative losses. Solid arrows mean data, dashed arrows mean weight updates/reuse; training-only content should be clearly bounded. Color must reinforce labels, not carry the distinction alone.

Small abstract window/vector icons can help. Do not depict fabricated restored pose examples as empirical corrections: verified source-pose reconstructions are unavailable. Any stick figure must be clearly schematic and must not imply measured successful side recovery.

## Symbols that can leave the graphic

Move `π_c`, `Render`, `Pose`, `T_m`, `N_n`, `E_δ`, `i∈{a,b}`, the full residual `e_i`, `H`, `c`, `τ_s`, `τ_t`, `sg`, `D`, and `1/(2D)` into the caption or existing equations. Replace them visually with projection, physical mirror, naming error, and centered/scaled scores. Normalization constants and temperatures should not occupy the central comparison.

Keep only labels that reduce ambiguity: original/edited states (`a`,`b` if helpful), observations versus clean references, endpoint versus delta, retained student encoder, readout fitting, and one-window inference. If a short formula uses score differences, define its symbols as the transformed scores, rather than silently changing what the implemented loss compares.

## Final redesign check

A reader should be able to answer in a few seconds: What are the two motion states? Which input is privileged? Where does the pair enter the loss? Which objective changes? Which encoder is reused? What does deployment require? The caption and method equation can then supply mathematical details without asking the diagram to serve as executable pseudocode.

## Final disposition: illustrated experimental workflow

**Accepted. No open material or minor methods finding remains.** This disposition concerns the latest three-step workflow with illustrative twelve-joint windows, fitting on 112 people, and evaluation on 14 development people. It supersedes the intervening encoder/EMA architecture candidate. The final rendered PNG, figure-generating source, caption and design provenance were inspected.

The latest user brief appropriately moves EMA, channel-score transformations, query support, physical-mirror mechanics and naming-corruption details into the methods. The figure now explains the experimental sequence instead of restating its implementation. Those omissions are acceptable at this level: no graphic element contradicts the detailed method or implies future-state dynamics.

### Semantics checked

- **The gait illustrations are schematic.** Both the graphic and caption say that the keypoints are illustrative. The source constructs twelve hand-designed 2D points for bilateral shoulders, elbows, wrists, hips, knees and ankles; no motion-capture or predicted-pose arrays are plotted. Stacked cards indicate a complete 128-frame window, not a three-frame input. The green leg identifies an illustrative edit, not a clinical side or measured successful restoration.
- **No false edit magnitude is reported.** The internal 28° drawing rotation changes the illustrative ankle location and is explicitly documented as a drawing choice. No numeric magnitude appears in the graphic or caption. It does not replace the study's nominal 3D intervention levels or assert a measured 2D response. The caption correctly identifies the study intervention as a synthetic 3D knee edit while marking the drawing as illustrative.
- **Privileged supervision is visible.** Noisy poses enter the student/predictor route; clean projected references supply teacher targets. Readout fitting is explicitly reference-supervised within the training panel. The caption's link to the feature-loss equation preserves the distinction between this high-level feature comparison and the precise centered/temperature-scaled score losses.
- **Endpoint and delta remain alternatives.** “Same base loss; separate runs” and “OR” prevent the false impression that the two auxiliaries are summed, share a final teacher, or define a single combined method. The original-to-edited arrow denotes a synthetic edit, not a prediction of state b from state a.
- **Training and development are distinct.** Fitting is labelled with 112 training people. Evaluation is labelled with 14 development people and three fitted seeds. The caption expressly says the development cohort is reused, so the graphic does not imply independent confirmation. Exact person-split inheritance remains in the methods and integrity appendix.
- **References enter development scoring only.** The observed window is the sole input to the frozen encoder and trained readout. The reference arrow enters the scoring box, not the model. The caption retains the direct-fitting exception: that control jointly adapts encoder and readout, unlike the illustrated frozen-encoder procedures.
- **Independent restoration and paired scoring are now separate.** The input is window a or b, restored independently. The final labels say “Paired asymmetry-change error,” “References a,b,” and “score the change from both windows.” The caption explicitly requires paired outputs and references for the response error, while knee-angle trajectories and post hoc naming are scored per window.
- **Outcome labels are exact.** “Knee-angle trajectory error” identifies the angular waveform outcome, avoiding confusion with Cartesian joint-position error. Left–right naming is visibly labelled post hoc. No zero-response trajectory, clinical validation, recovered pose example, or unperformed result is implied.

### Resolved refinements

The first workflow draft could suggest that all outcomes were scored independently for one window. The revised scoring labels and caption now expose the paired response requirement. The final “knee-angle trajectory” wording also resolves the remaining outcome-label ambiguity. The earlier architecture candidate's common-mask phrasing is superseded; the latest workflow makes no claim that the independently sampled artificial masks are identical.

The figure is visually readable at its 5.5 × 3.15 inch source size. Labels, boxes and arrows remain separate; color reinforces explicit endpoint/delta and edit labels. The short caption stays below 90 words. Full loss algebra is available through the live equation reference, rather than occupying the central visual flow. The final design-provenance asset hashes all match the inspected files.

### Final source bindings

| File | SHA-256 |
|---|---|
| `figures/method.png` | `2e7ba763009078a37f3d7ea94103e8af90e7b4b4c148327f701d2371170d222a` |
| `figures/method.pdf` | `f8d6ed4f2203ec9f097bdf6b988cecc5d1db0bf519c98224d0968afb53816c44` |
| `figures/method.svg` | `b3a75bd29a1c5770e6d9b0e217d645c67e3db2c1a39a37d58134d76d072c583a` |
| `figures/method-gray.png` | `508757ac04e693e83fb45e67e295b9ceb0b2ce53945a83448e9fa0ae01a43f0f` |
| `scripts/build_figures.py` | `06f42606c966d85fc7e47acf6115d5de53590ae29f3e3cf82e4d7d339dde90ba` |
| `paper-v08.tex` | `b726551c3e59666e29c0dbe2f6fefc5b55df1ba285a977cf41519bb9e98fe011` |
| `evidence/figure1-design-provenance.json` | `34156366559b8a7939a602496be81b82c9cfad1262397ebf7d78c127a7b5442d` |

This is a source/figure semantic and readability review. It performs no new training, statistical analysis, or validation of the hand-designed icons against actual participant motion. The parent handles the final full-document build and archive checks.
