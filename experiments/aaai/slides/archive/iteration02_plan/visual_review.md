# Review of the proposed second-iteration deck

The active deck contains 40 pages. It combines the proposed study with six new masking pages, numbered 14 through 19. Every PDF page was rendered with Poppler. All four contact sheets were inspected. The denser tables, mask grids and changed workflow pages were also inspected at full size. The four original shared diagrams and both new masking diagrams were inspected independently.

## Layout checks

The review checked title length, wrapped introductory text, table-row height, box padding, arrow placement, footer separation and ordinary reading order. Long titles were shortened on the masking pages. The mask grids group joint indices by anatomy and explicitly label the columns as four-frame patches. Their teal cells show the structured core, blue cells show uniform fill, and pale cells show positions that remain visible before validity checks. The anatomical example is explicitly one distal-leg draw, rather than the entire A policy.

The timing diagram uses one 48-frame source timeline with 12 four-frame tiles and three 32-frame context windows, starting at tiles 1, 2 and 3. The target interval is 8:40; the shifted donors are 4:36 and 12:44. It does not draw a synthetic gait trajectory or imply that a displaced limb forms a physically valid patient. The residual diagram keeps arrows clear of labels and states that retaining the baseline does not guarantee better test performance.

No text clipping, overlapping labels, crossed text arrows or out-of-page content was found in the inspected final pages. Pages 15 through 19 make the mask construction, matched comparison and gates legible without empirical curves or invented results. Pages 25 through 29 distinguish paired support, the compound timing-and-geometry challenge, the raw timing effect, separate score references and within-fold evaluation.

The automated [validation record](review/validation.json) reports 40 PDF pages, no out-of-page words, no broken local links, no Type 3 fonts and no embedded raster objects. The vector PDF uses embedded TrueType fonts. The link check covers the current README, notes and HTML, plus the archived deck's README, notes and HTML. The [mask-design record](review/mask_design_validation.json) checks the proposed construction algebra: 128 tokens, exactly 64 hidden, an eight-token structured core, 56 fill tokens and a one-half marginal hiding probability. It enumerates all 16 anatomical cores. This is an analytical proposal check, not execution of the future training sampler or scientific models.

## Scientific adversarial review

| Risk | Resolution in the deck |
|---|---|
| A proposal could look like a completed result. | Page labels separate measured draft01 values, read-only counts and proposed experiments. New notebooks and masks are explicitly unexecuted. |
| The 0.641 versus 0.541 result could be overstated. | Page 2 shows the paired interval crossing zero and the stronger calibrated full-error comparison. |
| A learned encoder could receive credit without evidence. | Page 3 shows random-coordinate 0.701 versus learned-coordinate 0.686 and the normal-development failures. |
| A standard mask could be presented as new gait methodology. | Page 14 identifies the motion-aware MAMP and S-JEPA defaults, the close SLiM precedent and the limits of action-recognition evidence. |
| Different mask budgets or joint frequencies could explain a difference. | Pages 15 and 16 specify U, J and A with exactly 64 hidden tokens and one-half token marginals. The notes distinguish matched marginals from matched difficulty or support. |
| Native-mask scores could conflate training and evaluation changes. | Page 17 compares all three arms within the unchanged U bank or the separately frozen distal-leg diagnostic bank. Its notes identify the A-versus-U contrast, the fixed seeds and bank-specific reference rules. |
| Changing the mask could silently change optimization or the second view. | Page 18 separates the correction-training and encoder-pretraining factors, retains the shared U-trained baseline and requires dedicated random streams plus the assigned second VICReg mask. |
| A masked hip or unavailable joint could leak into a derived feature. | Pages 16 and 19 state the conditional shared-reference caveat, mask-before-feature rule and required support audit. They do not claim independence from measurements used to build the declared shared reference. |
| A whole-joint tube could be discarded by the old target-history selector. | Page 19's notes require a declared own-leg context rule for the new training and diagnostic tasks while preserving historical primary U eligibility. |
| A weak local predictor or extra parameters could create a spurious relation. | Pages 20 through 23 retain the local prediction, add an own-leg capacity control and require known-answer checks. The primary contrast uses matched own-leg and both-leg corrections. |
| Cross-fold score scales could distort a pooled ranking. | Page 29 makes the primary AUROC a fixed pair-count-weighted within-fold mean and reports average precision per fold with prevalence. |
| An uninformative sparse context could imply timing independence. | Page 26 requires adequacy and headroom checks on the exact paired-support aligned inputs. |
| A failed relational gate could remove difficult sources. | Pages 19, 27 and 35 retain the fold with R = 0, so S = M/2 preserves its within-fold local-motion ranking. |
| The local-motion channel could become another prediction-error score. | Page 27's notes exclude local prediction error from M. M is fitted on fitting normals, while R uses the separate reserved normal references. |
| A timing shift could silently change the target or be called a pure phase intervention. | Pages 25 and 26 fix the target, body anchor and own observations; shift permitted hip-relative distal offsets; and require common donor and anchor support. They explicitly identify the compound timing-and-geometry change. |
| A phase estimate could leak the answer or erase the phenotype. | Page 24 separates descriptive phase from predictor inputs and preserves raw timing, amplitude and asymmetry. |
| Recovered normal sources could be called independent validation. | Pages 10 and 28 state that all 28 candidate normal sources were previously explored. |
| A changed endpoint could overwrite the historical primary endpoint. | Page 31 identifies the new focal primary as exploratory and requires the all-folder secondary using the same frozen scores. The historical endpoint is preserved. |
| Neurological motivation could become an unsupported diagnosis. | Pages 5 through 8 describe overlapping observable hypotheses and close prior art. The notes state that Q-PCI studied healthy adults and that myopathic is not one brain lesion. |
| An external dataset could be presented as a completed transfer result. | Pages 36 and 40 describe an optional early, bounded method-replication pilot. The notes preserve its two-hour parsing gate and one-working-day limit. No scientific external data were downloaded for this review. |

The [exact prior completed 40-page PDF](../draft01_completed/tutorial.pdf) remains unchanged. Its SHA-256 is `bb0502c207305ac9a5f2746c3af5dfaac7eda8c5d7805aa6007726f86fc317d8`. The current PDF's hash and empirical input hashes are recorded in [build_manifest.json](build_manifest.json). The two decks have the same page count but different purposes: the archive reports completed draft01 work, while the active deck explains the next iteration.

This review supports the readability and internal consistency of the planning artifacts. It does not establish novelty, efficacy, clinical validity or workshop acceptance. Those claims depend on the proposed experiments and the final manuscript review.
