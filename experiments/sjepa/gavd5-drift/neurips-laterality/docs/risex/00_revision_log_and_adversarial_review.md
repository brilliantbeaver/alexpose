# Seven-draft revision log and adversarial review

`codex:adversarial-review` was requested but is not installed in this environment. This record applies the same adversarial standard: every central claim was checked against `paper_v8.md`, the retained results summarized in `TUTORIAL.md`, and the available project evidence record. No new experiment was run.

| Version | Main change | Critique that prompted it | Improvement retained next |
|:--|:--|:--|:--|
| 01 | Condensed V8 into the RISEx headings. | Too much health and world-model framing for a robotics/AI one-pager. | One question, one outcome, and no deployment claim. |
| 02 | Reframed relevance around human-aware perception. | The first draft still implied that human gait alone establishes robotics relevance. | Treat robotics as a motivation and future use case. |
| 03 | Made the negative result quantitative. | The story lacked the evidence needed to distinguish a failed pretext task from a failed readout. | Report the matched initial control and correct-clip diagnostic together. |
| 04 | Added the 16-by-33 representation, source split, and uncertainty. | “Rigorous” was asserted without enough one-page evidence. | Name the preserved token grid, source-held-out unit, five seeds, and conditional intervals. |
| 05 | Removed low-value mask and reflection detail. | The page read like a compressed lab notebook. | Keep only the main result and one compact figure/table. |
| 06 | Hardened causal language and limits. | Preprocessing disagreement and readout confounding make a representation-wide claim unsound. | State the conclusion as applying to the tested training-and-readout pipeline. |
| 07 | Produced the submission candidate. | The prior draft still carried too much background for one page. | A reader can identify the task, control, result, and implication without the notebook history. |

## Adversarial findings applied to version 07

1. **The predictor diagnostic is not a common quality metric.** Each trained condition predicts its own EMA-teacher features. The final paper reports correct-clip preference as a diagnostic and does not rank methods by raw latent loss.
2. **The laterality target is not a clinical diagnosis.** Stroke, Parkinson's disease, cerebral palsy, and muscular disorders motivate interest in gait asymmetry, but V8 has no verified affected side or independent clinical target. The final paper says so.
3. **Source holdout limits direct leakage but not all dependence.** Repeated appearances of a person across videos cannot be ruled out. The final paper calls the held-out unit a source video rather than a participant.
4. **The target/readout path can explain the deficit.** Preparation changes the target; expanded summaries combine several information sources; penalties frequently hit the search boundary. The final conclusion is limited to the tested pipeline.
5. **RISEx format cannot carry a full methods paper.** The final candidate therefore makes no claim of a general robotics solution. It is a research snapshot suitable only if authors accept the archival tradeoff and confirm the venue policy.

## Final pre-submission checks

- Replace the author placeholders in the RISEx Word template; RISEx's public call does not state that review is double-blind.
- Fit version 07 into the supplied, unmodified template. The template specifies one letter-sized, two-column page in 11-point Times New Roman and calls for the headings used here.
- Confirm the live submission mechanism: the public call says OpenReview, whereas the downloaded template still mentions a Google Form.
- Confirm whether concurrent or planned archival submissions are permitted.
- Verify the cited paper metadata and all numbers after final layout. Do not release videos, poses, embeddings, predictions, or checkpoints without the appropriate data-use approval.

