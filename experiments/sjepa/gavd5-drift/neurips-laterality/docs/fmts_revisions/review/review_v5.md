# FMTS V5 adversarial review

| Perspective | Objection and severity | Evidence | Correction in V6 | Residual limitation |
|---|---|---|---|---|
| Temporal researcher | Major: losing a low-level speed quantity may be acceptable for action recognition, so why is this a temporal-model result? | Training predicts learned features; evaluation deliberately chooses a different system observable. | State that the test concerns suitability for the chosen gait measurement, rather than universal preservation of all details. | No claim that JEPA should preserve every observable or that this is a general JEPA failure. |
| Evidence auditor | Moderate: reflection augmentation and reflection loss need their distinct recipes even after chronology is removed. | Augmentation summary uses p=0.5, scheduled teacher EMA and four inner folds; explicit loss only synthetic. | Preserve recipe, evidence type and extra-pass cost in one appendix paragraph. | No real-data reflection-loss effect estimate. |
| Statistical reviewer | Moderate: provenance details consume space without making bootstrap reproducible from aggregates. | Raw predictions absent; numerical supplement contains seed aggregates and retained intervals. | State the limitation once and remove local execution counts from manuscript prose. | Source resampling cannot be rerun with aggregate-only supplement. |
| Editor | Major: Markdown References is only a working placeholder despite a complete PDF bibliography. | V1-V5 use Pandoc citation keys and BibTeX. | Materialize complete numbered Markdown references with a private key map, keeping natbib/BibTeX in PDF. Shorten appendix duplication. | No requirement inferred for a main-track checklist: FMTS CFP does not request one. |

V6 retains two short appendices rather than a notebook catalogue. Detailed future controls and provenance remain in the FMTS README record.
