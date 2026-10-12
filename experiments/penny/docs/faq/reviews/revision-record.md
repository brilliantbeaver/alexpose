# FAQ review and revision record

This record concerns the new illustrated FAQ, not modifications to the original GAVD3 experiments. The scientific review used notebook code and retained real artifacts. Training, pose extraction, and downstream experiments were not rerun. A deterministic preprocessing fixture checked one code property; its outcome is not presented as a research performance result.

## Review process

The evidence review inspected both `gavd3/cache/artifacts/real` and the ignored `gavd3/work/artifacts/real`. A preliminary search missed the latter directory. The recursive check corrected that omission before drafting the results narrative, and all later results were included with their method boundaries. Empty notebook outputs were not treated as proof that an experiment had never run.

The draft received separately assigned [factual](faq-factual-review.md) and [comprehension](faq-comprehension-review.md) reviews. Those reviewers used fresh tasks but belong to the same model family as the author; they do not constitute model diversity.

The official [`codex:adversarial-review` command](https://github.com/openai/codex-plugin-cc/blob/main/plugins/codex/commands/adversarial-review.md) ran through its installed companion runtime against a frozen temporary Git repository. The evidence and source notebooks formed its baseline, and only the FAQ draft was the review diff. The original workspace and its unrelated changes were excluded. The [recorded target](draft-review-target.json) preserves the snapshot commits and draft fingerprint. The [full result](codex-adversarial-draft.json) returned `approve` with no material findings; this is a review of prose and inference, not a rendered-document approval or independent experimental replication.

The requested review instructions and prompt were downloaded from the official repository, with [provenance](skill-sources/provenance.json). The additional [codex-adversarial-review skill](https://github.com/aarontzeng/dev-lead/tree/master/skills/codex-adversarial-review) was installed and read, including its framing and runtime guidance. Its cross-family recommendation led to a separate Claude CLI attempt. That configured service returned connection-refused errors before review; the [attempt record](claude-adversarial-draft.md) explicitly reports it as unavailable. No cross-family approval is claimed. The user's explicit instruction to revise authorized corrections after the review-only passes.

The final artifact checks cover the generated Word document, actual SVG media, original image bytes, alternative text, bookmarks, page numbering, heading placement, and Markdown-to-Word text agreement. The Word file was rendered through LibreOffice for visual inspection. These checks assess the documents, rather than clinical or model validity.

## Factual review dispositions

| Issue | Correction | Disposition |
|---|---|---|
| FR-01: input coordinate frame | Question 17 now explains remapping crop detections to full-frame normalized coordinates; Question 36 distinguishes crop quality from coordinate origin. | Addressed. |
| FR-02: image-plane measurement | Question 36 identifies camera motion, normalized-axis aspect ratio, and the percentile span used for knee excursion. | Addressed. |
| FR-03: phase protocol | Question 37 separates the temporal-segment, five-fold sequence-grouped phase reader from the pooled, leave-one-sequence-out scalar probes. | Addressed. |
| FR-04: missing uncertainty marks | The identity SVG now includes the recorded standard-deviation whiskers, and its caption identifies their three-split basis. | Addressed. |
| FR-05: omitted graphical control | The gait-probe SVG now includes the raw-coordinate column alongside S-JEPA and missingness. | Addressed. |
| FR-06: anatomy graphic mismatch | Figure 5 now uses an actual landmark diagram with regions linked to hypotheses. | Addressed. |
| FR-07: independence terminology | Notebook 01 is described as counting distinct sources. Participant independence remains an additional requirement in Question 30. | Addressed. |
| FR-08: provisional links | The missing census target was replaced with the preserved source-count table; the Word file, pose illustration, and this revision record were created. All relative targets receive a final check. | Addressed. |

## Comprehension review dispositions

| Issue | Correction | Disposition |
|---|---|---|
| C01: coordinate explanation | The input origin and proxy limitations were corrected with the same changes as FR-01. | Addressed. |
| C02: null as measurable improvement | Question 3 now frames pretraining as improving recovery or prediction relative to controls on independent sources, with explicit retrospective status. | Addressed. |
| C03: classifier and probe definitions | Question 15 explains decision trees, aggregation, and the weighted-sum linear readout before reporting scores. | Addressed. |
| C04: future dependence example | Question 40 explains how later body-width observations affect earlier prepared coordinates, states that the teacher sees the complete clip, and distinguishes simultaneous forecasts from recursive rollout. | Addressed. |
| C05: cadence and target quality | Question 36 defines step cadence separately from ankle cycles. Question 37 identifies unreliable pose targets as an alternative explanation for weak readouts. | Addressed. |
| C06: anatomical vocabulary | Question 20 translates bilateral, proximal, and flexion terms and defines proxy measurements. The suggested legacy-feature example was omitted because it would add an unrelated feature pipeline to this basic explanation; the underlying mapping caveat remains explicit. | Addressed by an equivalent explanation. |
| C07: F1 vocabulary | Question 26 gives the F1 expression and its practical interpretation, followed by equal category weighting. | Addressed. |

## Additional evidence-driven corrections

The retained artifacts supplied the corrected 33.3% exact-split and 48.3% all-sequence majority values. The grouped 49.0% comparator was checked against training-fold modal-label code, not inferred from test labels. The original main S-JEPA run and the retrained mask reference are distinguished.

The FAQ qualifies the source-grouped result because one fold removes normal from classifier training while its encoder has already seen the held-out normal inputs. It describes source-ID results without assigning a causal fraction of classification performance to memorization. It treats single-seed mask differences as absence of demonstrated benefit, rather than equivalence.

The scalar probe values were taken from retained tables rather than stale summaries. The ablation cadence target is identified as cycles per resized frame, distinct from the original-duration scalar probe. The forecast's `circular_correlation` field is described as its implemented mean cosine phase alignment. Forecasting comparisons state the model's in-sample normal evaluation, the baseline's query-excluding and joint-pooled construction, and the full-clip preprocessing dependence.

## Remaining scientific limits

Multiple independent normal sources, fold-specific encoder pretraining, participant metadata, repeated matched evaluation, and independently measured gait targets remain necessary for broader generalization or physical claims. Future prediction also needs past-only preprocessing and equally exposed held-out controls. The document does not treat these unresolved research requirements as completed corrections to the experiments.
