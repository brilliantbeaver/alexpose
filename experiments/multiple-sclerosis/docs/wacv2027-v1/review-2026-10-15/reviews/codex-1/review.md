1. **HOLDS — numerical consistency against retained outputs.** Notebook 06 records RF `0.325184 / 0.320940` and S-JEPA `0.456624 / 0.397034` for source/clip weighting (`06_capstone_rf_vs_sjepa.ipynb:1010`, `:1011`). The replacement tables, fold scores, confusion percentages and bootstrap endpoints agree with the October artifacts. Historical RF is disclosed explicitly: “the earlier RF is slightly higher before rounding” under clip weighting (`revision/02-overleaf-replacements.md:156`). Comparing the two saved `oof.json` files shows exactly 12 RF-label changes and no S-JEPA-label changes. The diagnostic means appear as `3.858368` and `1.022220` in saved notebook output (`03_sjepa_model_and_pretrain_normal.ipynb:1132`, `:1139`). These are saved summaries, not recovered per-update histories. I accept the supplied host verification of checkpoint identity; I did not reproduce it.

2. **HOLDS — the positive claim remains bounded.** “The evidence supports useful **trained representations**” is immediately qualified by the missing untrained-encoder comparison (`revision/01-priority-revisions.md:9`). Mean pose’s stronger clip-weighted result is retained: “0.410 versus 0.397” (`revision/02-overleaf-replacements.md:133`). Masking benefit remains untested: “a separate comparison would be needed” (`revision/02-overleaf-replacements.md:67`). The text supports an observed trained-feature pipeline result without attributing its advantage causally to pretraining, temporal order or anatomical masking.

3. **HOLDS — substantive methods match the inspected implementation.** Training sizes agree with the retained split summaries; fitting and validation selection remain distinct. The encoder loop takes `dataset[int(i)][0]`, excluding labels (`sjepa/train_v2.py:180`), while the classifier fits labels (`sjepa/full_experiment.py:64`). Fresh optimizer/center creation and seeded streams support the restarted continuation description (`sjepa/train_v2.py:150`, `:154`, `:169`). Readout uses full inputs and averages selected output positions: `feats = self.target_encoder(x_full)` followed by `feats[:, idx, :].mean(dim=1)` (`sjepa/models.py:308`, `:312`). Mask frequencies use separate denominators, including `target_token_count / (n_masks * T)` (`sjepa/masking_v2.py:229`). Source weighting is `1.0 / counts[r.source_id]` (`sjepa/full_experiment.py:69`), and the registry explicitly says `"source_video_not_participant"` (`artifacts/eval/full-v1/fold_registry.json:4`). The Figure 1 wording discrepancy is finding 2.

4. **BROKEN — the guide has delivery and instruction defects.** The evidence links resolve incorrectly from `revision/`, and two smaller instruction issues remain below. R15’s substantive boundary is now precise: “Preserve the intervening MediaPipe extraction, cleaning description, equation (1), Table 1 and their citations” (`revision/02-overleaf-replacements.md:266`). R3b explicitly retains MaskFeat’s citation (`:51`). Numerical text/table/figure agreement holds on inspection. Compiled layout, final cross-references and bibliography resolution are **NOT REACHED**.

5. **HOLDS — as a textual assessment.** The main narrative explains the experiment and its outcomes without repeatedly organizing them around statistical non-significance. It defines macro-F1 before explaining equal-source weighting (`revision/02-overleaf-replacements.md:119`) and reports concrete failures, including “13 MS clips are assigned to PD” (`:161`). Detailed uncertainty remains available in R14. This is an editorial judgment, not a reader-comprehension test.

6. **NOT REACHED — complete provenance verification.** The statement “The recorded check passed” (`revision/03-evidence-and-submission.md:45`) has a retained supporting record: `"all_saved_metrics_match": true` (`docs/wacv2007/review-2026-10-15/evidence/results-verification.json:2127`). However, the verification script, author-kit/source-page snapshots and referenced review-disposition document are absent from this reviewed tree. Their implementation or contents cannot be independently confirmed here. Venue requirements and literature checks are recorded in the saved venue audit (`docs/wacv2007/review-2026-10-15/evidence/paper-and-venue-audit.md:15`, `:19`, `:67`); I did not recheck live sources. Absence from this snapshot does not establish that the reported earlier work never occurred.

The findings are:

1. **Should-fix — introduced; high confidence. Broken evidence entry points in the reviewed commit.**  
   Locations: `revision/03-evidence-and-submission.md:3`, `:45`, `:62`; related promises at `revision/02-overleaf-replacements.md:3` and `revision/03-evidence-and-submission.md:76`.

   Opening `[evidence/](evidence/)` from the new companion resolves to `revision/evidence/`, but the supplied audits and verification JSON reside under `docs/wacv2007/review-2026-10-15/evidence/`. The results and venue links consequently fail. The advertised `latex/`, verification script and review-disposition targets also have no entries in the target commit.

   **Consequence:** authors following the guide cannot reach its supporting evidence through the supplied links or obtain the advertised companion files. Inline replacement text remains usable, limiting the reach of this defect. Rebuilding the manuscript does not check these Markdown links.

   **Fix:** point links to their actual committed locations, using `../docs/wacv2007/review-2026-10-15/evidence/…` where appropriate. Supply the other companions or explicitly identify them as separately held materials rather than available links.

2. **Nit — introduced; high confidence. Figure 1 describes executed training as optional.**  
   Location: `revision/02-overleaf-replacements.md:9`.

   Following the instruction produces “800 updates; optional 400-update continuation.” The actual procedure executes both stages in every fold: `for stage, budget in (("ssl", updates), ("continued", more_updates))` (`sjepa/full_experiment.py:132`). Validation selects which stage supplies predictions; it does not decide whether continuation runs.

   **Consequence:** the schematic can understate the executed training budget. The explicit account at `revision/02-overleaf-replacements.md:80` mitigates this.

   **Fix:** use “800 updates + restarted 400-update continuation; select stage using validation.”

3. **Nit — introduced; medium confidence. Moving only equation (2) loses its explicit DINO attribution.**  
   Locations: `revision/02-overleaf-replacements.md:72`, `:78`.

   R5 replaces §4.3 and instructs authors to move the equation, settings and diagnostic definition. The original equation’s introductory sentence explicitly attributes centering and sharpening to “DINO [35]” (`docs/wacv2007/review-2026-10-15/working/original-paper.txt:199`). Neither the replacement paragraph nor R14 carries that attribution.

   **Consequence:** literal application removes the method-specific credit, although general background citations and S-JEPA attribution survive elsewhere.

   **Fix:** explicitly move the equation’s attribution sentence and citation with it, or retain the original DINO citation in R5.

No added tests exercise manuscript assembly, citation preservation or reader comprehension. No retained comparison isolates pretraining, temporal-order or masking effects, and no check establishes verified diagnoses, participant independence or clinical validity. These are limits on interpretation, not requirements for new experiments before reporting the bounded observational result.

Metric agreement verifies calculations from fixed predictions; it does not reproduce feature extraction, classifier fitting or neural training. R14 states this boundary: “They do not include changes from retraining, checkpoint selection or alternative source partitions” (`revision/02-overleaf-replacements.md:254`).

A historical check passed for an insufficient reason: **15 nonconstant RF columns** remained true despite two columns duplicating the same ankle measurement (`docs/wacv2007/review-2026-10-15/evidence/results-audit.md:23`). Likewise, window-level mask coverage cannot establish balanced coverage at every time position; the notebook now explicitly warns about this (`02_anatomical_mask_and_tokenization.ipynb:773`). Neither limitation demonstrates a current false pass of the numerical scoring check.

Evidence anchors:

- `revision/01-priority-revisions.md`: **55 lines**. Verbatim final line:

  ```text
  The supplied replacement sections are intended for an anonymous regular paper. Apply them together, rebuild with the official review template, and verify the final page count and citation keys. [03-evidence-and-submission.md](03-evidence-and-submission.md) records the current venue requirements and numerical evidence.
  ```

- `revision/02-overleaf-replacements.md`: **286 lines**. Final line: `""` — an empty line. Line 285 is the closing triple-backtick fence.

- `revision/03-evidence-and-submission.md`: **76 lines**. Verbatim final line:

  ```text
  Three parallel audits separately examined notebooks/methods, numerical results and the paper/venue. The resulting replacement text then underwent a fresh Codex adversarial review. [04-review-disposition.md](04-review-disposition.md) records the review target, findings and fixes. The independent contexts provide additional scrutiny within the available Codex model family; they are not model-family diversity or independent scientific replication.
  ```

Verdict: needs-attention