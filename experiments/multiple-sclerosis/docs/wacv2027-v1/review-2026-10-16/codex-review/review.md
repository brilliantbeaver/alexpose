Notation: **main** is `docs/wacv2027-v1/wacv-v1-revision.tex`; **R** is `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20261009T060955236580Z`. Notebook cells below are one-based, using the supplied text exports.

1. **HOLDS — reported numbers; NOT REACHED — rendered figure contents.**

   Main lines 78–82 match the recorded inventory: raw clips 26/30/35, usable clips 24/29/35, and sources 16/13/12 for Normal/MS/PD. Notebook 00, cell 9, prints `"total clips: 91 | total sources: 41"`; notebook 01, cell 14, records the three exclusions and usable counts. Applying the inspected windowing rule, `"starts = list(range(0, T - window + 1, stride))"` (`sjepa/data.py:279`), to the registry’s saved frame counts confirms 59/171/128 windows and 5/0/10 padded clips.

   Main lines 194–200 agree with the raw result JSON:

   | Method | Source F1 | Clip F1 | Correct clips |
   |---|---:|---:|---:|
   | S-JEPA | 0.457 | 0.397 | 36 |
   | Corrected RF | 0.325 | 0.321 | 30 |
   | Mean pose | 0.452 | 0.410 | 37 |
   | Visibility | 0.318 | 0.318 | 29 |
   | Majority | 0.259 | 0.202 | 23 |
   | Earlier RF | 0.411 | 0.397 | 37 |

   For example, R/results.json:303 records `"macro_f1": 0.4566241590955206`; line 187 records RF’s `0.32518410410355597`. Notebook 06, cell 14, prints the corresponding current scores. The September run’s results.json:139,187 records RF values `0.397315739713076` and `0.41126291036663587`. Thus the manuscript’s differences **0.131** and **0.045** correctly use unrounded scores. The earlier RF’s clip F1 is slightly higher than S-JEPA’s despite both rounding to 0.397.

   Direct comparison of the two `oof.json` files shows exactly 12 changed `pred_rf` entries and no changed S-JEPA predictions. Current extraction assigns `"left_ankle_range=safe_get(left_ankle_stats, \"range\")"` (`ambient/classification/features.py:912`). The host’s feature comparison additionally records `"all_other_feature_values_identical": true` at evidence/results-verification.json:2847.

   Main line 210’s fold comparisons match the saved figure values: S-JEPA exceeds corrected RF in folds 0–2; mean pose exceeds S-JEPA in folds 0, 1, 3 and 4. Its endpoints are `0.21726707059235983` and `0.7795975953870691`.

   Main line 212’s percentages match R/results.json:191–203,307–319. Its raw S-JEPA confusion rows are `[10,6,8]`, `[8,8,13]`, `[10,7,18]` at lines 276–291.

   `figure-values.json:3–78` matches the current confusion counts and fold scores; `build_figures.py:55–61` uses `"pct = 100 * cm / cm.sum(axis=1, keepdims=True)"` and a common 0–100 scale. The rendered PDFs are absent from this supplied tree, so their actual contents remain unverified.

2. **HOLDS — 24 fitting sources, with additional validation sources.**

   Main line 148 explicitly says `"fitting and stage selection together use 32--33 development sources"`. Notebook 02, cell 7, prints all five partition rows, matching main lines 157–161: training clips 51/58/54/53/45 from 24 sources each; validation sources 8/9/9/9/9.

   `sjepa/full_experiment.py:98–100` rejects training membership that differs from the registry. Lines 169–183 fit the probe on training embeddings, choose the stage using validation, and only then extract test embeddings. The selection expression is `"continued" if validation_scores["continued"] > validation_scores["ssl"] else "ssl"`.

   Source exclusion does not establish participant exclusion. The registry explicitly records `"grouping": "source_video_not_participant"` at line 4, and main line 145 states that participant separation cannot be established.

3. **HOLDS — active architecture and experiment description, with specific evidence gaps below.**

   R/results.json:7–20 records `"frame_group": 4`, `"encoder_dim": 96`, `"encoder_depth": 3`, `"encoder_heads": 4`, `"predictor_depth": 2`, and batch size 32. These agree with main lines 100–102. `sjepa/tokenizer.py:47–50` groups frames and adds separate spatial/temporal embeddings; `sjepa/models.py:128–133` masks attention keys and replaces hidden outputs before prediction.

   Main lines 120–124 describe the active label-free training path, rather than the legacy notebook filenames. The loop reads `dataset[int(i)][0]` (`sjepa/train_v2.py:180`) and optimizes `"loss = ce.masked_batch(predicted, target, tgt)"` at line 220. No VICReg term enters that path. Notebook 04, cells 11 and 14, executes continuation and validation selection.

   Mask regions, overlap weighting, spans and context guarantee agree with `sjepa/masking_v2.py:43–50,101–105,135–156`. Readout agrees with `full_experiment.py:26–35` and `models.py:308–312`: the complete input enters the target encoder, then selected **outputs** are averaged.

   Cleaning, interpolation without a gap limit, normalization, padding and omitted trailing windows agree with `sjepa/data.py:130–182,266–282` and notebook 01, cells 10 and 14. RF, mean-pose and visibility feature construction agree with `sjepa/classical.py:34–50,80–94`, `ambient/classification/features.py:874–915`, and `full_experiment.py:77–88`.

4. **HOLDS as a bounded optimization/nonconstant-output description; stronger learning attribution is NOT REACHED.**

   Notebook 03, cell 19, prints loss means `"3.858368"` and `"1.022220"`, mean effective ranks `"8.132109"` and `"10.939291"`, and endpoint rank `"12.134889"`. Main line 215 correctly distinguishes the loss averages from the endpoint rank.

   The loss uses `"F.softmax((target - self.center) / self.tau_target, dim=-1)"` and updates the center every batch (`sjepa/losses.py:87–92`); teacher weights also change (`train_v2.py:227–231`). Consequently, decreasing cross-entropy does not isolate improved prediction against fixed targets: both target entropy and prediction mismatch can change.

   Effective rank centers the sampled batch and normalizes its singular values (`train_v2.py:92–101`). It measures variation in all-token pooled teacher embeddings, not clinical separation or the exact classifier readout. The nonzero feature spread supports the narrower conclusion that these sampled embeddings are not constant.

   “Fits the task” is defensible shorthand for this optimization evidence, but more literal wording would avoid implying a fixed-target learning assessment. Main line 215 explicitly says the observations `"do not measure how much classification improved relative to an untrained encoder"`; line 226 repeats that boundary. This warrants a precision refinement, not a requirement for another experiment.

   R/results.json:43–128 also confirms lower continuation endpoint losses in every fold and selection of continuation only in folds 3–4.

5. **HOLDS — the abstract and conclusion make descriptive pipeline claims.**

   Main line 19 expressly says `"further controls are needed to isolate the contribution of pretraining and temporal information"`. Line 239 reports features useful for `"gait-video label prediction"` and calls for comparisons with untrained encoders and clinically verified recordings.

   The numerical “exceeds” claims describe the retained predictions. They do not claim statistical superiority, and supplementary lines 126–128 explicitly discuss intervals including either ranking and uncertainty conditional on fixed predictions. The manuscript also acknowledges the pretrained pose front end at main line 231. I found no abstract/conclusion claim of demonstrated clinical validity or an isolated causal benefit of pretraining.

6. **BROKEN — line 229 turns a visibility score into established evidence of signal.**

   The measured result is supported: R/results.json:419 records `"macro_f1": 0.317564627924763`, and notebook 06, cell 14, prints `0.317565`. The input is exactly `"np.concatenate([vis.mean(0), vis.std(0)])"` (`full_experiment.py:82`).

   But main line 229 says this score `"shows that detector outputs contain label-associated information"`. The saved evaluation calculates classification metrics; it does not establish that this visibility result differs from a label-independent reference. The training-clip majority comparator is not such a reference. A label-independent feature pipeline could produce this finite-sample score while passing the inspected checks.

   The following qualification—unknown gait versus recording origin—does not resolve whether signal has been established. This is a wording correction; it does not require adding experiments. It also does not establish that S-JEPA relies on visibility.

7. **NOT REACHED — historical curation and provenance assertions.**

   Main line 58’s search process, assignment from video titles, manual walking-segment selection and cropping history cannot be independently established from these notebooks and manifests. The registry records filenames, labels, source IDs and cache fingerprints, not original titles, crop decisions or clinical records. These claims are unsupported here, not demonstrated false.

   Likewise, the exact 5,000-mask result at main line 105 is not present in the inspected evidence. Notebook 02, cell 14, instead executes 512 masks and prints `"mean target fraction 0.63"`. That is compatible with 63.2%, but does not verify the stated 5,000-mask calculation.

8. **HOLDS — implementation checks have narrower guarantees than scientific controls.**

   No tests are added or edited by this range; it adds two manuscript files.

   Several existing checks could pass for the wrong *scientific* reason:

   - `full_experiment.py:106–108` checks `"np.isfinite(values).all()"`. Constant embeddings with finite zero rank could pass; this guards numerical execution, not successful learning.
   - Mask coverage uses `"(~m).any(axis=0)"` (`masking_v2.py:220`). A joint visible only rarely within windows can pass. Notebook 02, cell 14, simultaneously reports minimum window coverage `0.74` and middle-block hip visibility as low as `7.0%`.
   - Registry disjointness compares recorded source identifiers (`splits.py:163–166`). It cannot detect the same participant or duplicated footage under different source IDs.
   - OOF validation checks attribution and valid predicted labels (`full_experiment.py:208–223`). Correctly attributed chance predictions pass.

   No saved controlled comparison isolates pretraining gain, temporal-order benefit, anatomical-mask benefit, reflection effects or readout-mask sensitivity. Main lines 109, 122, 130, 226 and 236 already acknowledge these boundaries; their absence is not itself a reporting defect.

Evidence gate:

| File read | Line count | Verbatim last line |
|---|---:|---|
| `docs/wacv2027-v1/wacv-v1-revision.tex` | 245 | `\end{document}` |
| `sjepa/train_v2.py` | 345 | `    return ckpt` |
| R/`results.json` | 750 | `}` |

Both manuscript files are newly added and have no BASE counterpart.

Concrete findings:

1. **should-fix — introduced; high confidence — `docs/wacv2027-v1/wacv-v1-revision.tex:229`.**  
   **Misstatement:** A visibility-only F1 of 0.318 “shows” label-associated information.  
   **Evidence:** Notebook 06, cell 14; R/results.json:415–419; `sjepa/full_experiment.py:77–88,208–231`. These establish the score and feature construction, without a signal-versus-chance assessment. The qualification about signal origin does not address this inference.  
   **Replacement:** “The visibility-only classifier achieves source-weighted macro-F1 of 0.318. This descriptive result does not establish reliable label prediction or distinguish gait-related from recording-related cues.”

2. **nit — introduced; optional precision refinement — `docs/wacv2027-v1/wacv-v1-revision.tex:215`, repeated at line 226.**  
   **Imprecision:** “Fits” can suggest an assessed fixed prediction task, whereas the recorded cross-entropy uses evolving, centered teacher targets. The nearby limitations prevent this from becoming a material pretraining-benefit claim.  
   **Evidence:** Notebook 03, cell 19; `sjepa/losses.py:85–92`; `sjepa/train_v2.py:227–239`.  
   **Replacement:** “Recorded training cross-entropy decreased while sampled target-encoder embeddings retained variation.”

Remaining validation gaps: original curation and source provenance; rendered figure contents; direct checkpoint/feature-array payload comparisons, whose binary payloads are absent from this snapshot; the exact 5,000-mask calculation; and the explicitly unperformed scientific controls listed above. No tests, scripts, training or builds were run.

Verdict: needs-attention