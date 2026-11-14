# Review findings and resolution

The review supports a positive, bounded account of the experiment: the trained S-JEPA pipeline scores above the corrected handcrafted Random Forest on the retained held-out predictions. The replacement text preserves the strong mean-pose result and does not attribute the difference specifically to pretraining, temporal order or the anatomical mask. All numerical corrections refer to one coherent October execution, with the earlier RF result disclosed separately.

This document records the scrutiny applied to that account and the limits of the checks. The final delivery is under `docs/wacv2027-v1/review-2026-10-15`, following the author's updated instruction.

## Independent audit assignments

Three parallel agents examined different evidence before the replacement text was finalized:

| Assignment | Material examined | Main consequence for the revision |
|---|---|---|
| [Notebook and methods audit](evidence/notebook-methods-audit.md) | All seven notebooks, all 124 cells, saved outputs and relevant visualizations, with the active implementation traced in source code. | Corrected the masking denominator and clarified training, continuation, readout, sampling cadence and padding. Notebook filenames were not treated as descriptions of the executed procedure. |
| [Results audit](evidence/results-audit.md) | Current and historical predictions, registries, feature caches, provenance and retained model artifacts. | Identified the duplicated ankle feature in the old RF, recalculated scores independently, and separated repeated execution from independent replication. |
| [Paper and venue audit](evidence/paper-and-venue-audit.md) | All nine original PDF pages, figures and tables, the live workshop call, official author kit and primary literature. | Established exact replacement locations, submission requirements, closer related work and the limits of pose-validation citations. |

The host reconciled the audits and ran [the numerical verification script](evidence/verify_results.py). The resulting [machine-readable record](evidence/results-verification.json) confirms agreement with the saved metrics. A separate [mask and training-history check](evidence/mask-and-history-verification.json) verifies the mask percentages and connects the notebook's fold-0 diagnostics to the current capstone checkpoint. These checks use existing artifacts; no encoder or classifier was trained for this review.

## Codex adversarial review

The [requested review skill](/Users/pmui/.codex/skills/codex-adversarial-review/SKILL.md) was applied through a fresh, read-only Codex CLI context using `gpt-6-astra` at `xhigh` reasoning effort. The skill's helper scripts were absent from the installation, so the host used an isolated Git snapshot and SHA-256 checks to preserve the same target-integrity requirements. The [manifest](reviews/codex-1/manifest.json) records the base, reviewed commit and 43 file hashes. [Post-run verification](reviews/codex-1/post-run-verification.json) confirms that the commit, working tree and all recorded hashes remained unchanged.

The initial invocation failed authentication when user configuration was disabled; it produced no review. The retry used the normal configured authentication and completed successfully, without changing credentials. Both the failed [initial log](reviews/codex-1/review.log) and successful [review log](reviews/codex-1/retry.log) are retained. The latter confirms the model and reasoning effort actually used.

The [unaltered adversarial report](reviews/codex-1/review.md) ends with **`Verdict: needs-attention`**. It found that the numerical claims, interpretation and substantive methods held within their stated boundaries, and raised three delivery or instruction issues. The final files incorporate the resolutions below; the raw report has not been rewritten to imply that the reviewer approved a later version.

| Finding | Verification and disposition |
|---|---|
| Evidence links and some companion files were unavailable from the review snapshot's `revision/` directory. | **Valid for the snapshot, not the delivered directory.** The host had copied the three proposed documents into an artificial `revision/` folder while retaining evidence elsewhere and omitting several companions. This packaging defect limited the reviewer. The actual delivery places documents, evidence, LaTeX fragments and this disposition together. Relative links were checked against that final directory; redirecting them to the artificial snapshot locations would have broken the delivered package. |
| Figure 1 called the 400-update continuation optional although every fold executed it. | **Accepted and corrected.** The guide now specifies “800 updates; 400-update continuation tested,” states that optimization restarts, and explains in R5 that validation selects between the two executed stages. It no longer suggests that some folds skipped continuation. |
| Moving equation (2) could remove its original DINO attribution. | **Accepted and corrected.** The guide explicitly instructs the author to carry the attribution and citation with the equation. R5 also retains `\cite{dino}` beside the centering/sharpening description. |

The snapshot did not include the verification script, raw feature caches or checkpoints, downloaded venue pages, compiled preview or final disposition. The CLI reviewer therefore marked some provenance and layout checks **not reached**. Those checks were performed by the host and audit agents on the actual workspace, not by the CLI reviewer. The report also correctly separates recomputation from fixed predictions from reproduction of model training.

The available agents and CLI reviewer are from the Codex model family. Their separate contexts provide additional scrutiny, but do not meet the skill's preferred review by a different model family and do not constitute independent scientific replication. No separate model-family approval is claimed.

## Final editorial and delivery checks

A fresh agent reviewed the actual delivery directory after the main scientific corrections. Its [recorded assessment](reviews/final-editorial-check.md) found no additional material issue in the sample-size claims, related work, preservation instructions or interpretation. It confirmed that all 19 LaTeX fragments match their corresponding guide blocks. Its continuation-label concern is covered by the correction above.

The host also verified the final relative links, the fragment-to-guide agreement and the unchanged checksum of the original PDF. These results are recorded in [final-delivery-verification.json](evidence/final-delivery-verification.json).

The replacement prose, table and numerical figure were compiled in an internal preview using the official WACV review style. The final six-page preview has no unresolved references or overfull/underfull boxes, and every rendered page was inspected. The TeX engine reports an encoding warning in an installed `lineno.sty` comment; no corresponding rendering defect was observed. The preview includes 18 fragments; R16 supplies captions for the original illustrations and is checked as text. It omits unchanged manuscript passages and is explicitly labeled as an incomplete layout check. Its page count is **not** the page count of a revised submission, and it does not verify the author's complete Overleaf bibliography or final figure placement.

The original draft and existing Overleaf package were preserved. The temporary alias in the earlier output folder was removed after the CLI report finished; all new review outputs are now in the requested folder.

## What remains for manuscript assembly

The requested review and replacement passages are complete. Applying them to the original Overleaf project still requires the author to merge citation keys, update the original schematic labels, move the detailed statistical and training material to the supplement, and rebuild the complete anonymous manuscript. Exact instructions accompany the passages. The final content-page count must be checked on that assembled manuscript.

The source records do not establish verified diagnoses, distinct participants across source videos, permissions for secondary use, or an ethics determination. The revision identifies those documentation needs without inventing their answers. Additional experiments would be needed to establish a causal pretraining benefit or improved clinical performance; those stronger claims are not made in the supplied text.
