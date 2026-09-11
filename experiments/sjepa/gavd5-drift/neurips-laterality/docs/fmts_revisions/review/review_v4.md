# FMTS V4 adversarial review

| Perspective | Objection and severity | Evidence | Correction in V5 | Residual limitation |
|---|---|---|---|---|
| Temporal researcher | Moderate: the schematic could be mistaken for future gait prediction or a real pose example. | All drawn poses are hand-built schematics; the model completes hidden positions. | Label schematic poses, original time, index resizing and two-sided context at the point of use. | No observed-to-future gait demonstration is claimed. |
| Evidence auditor | Major: the full-view branch omitted its projector in the graphic. | motion_structured_training.py encodes both full views, pools gait joints, then projects before VICReg. | Add projector, clarify twelve-joint pool includes hips, and keep y entirely outside encoder supervision. | No ablation isolates the full-view regularizer. |
| Statistical reviewer | Moderate: result-plot dots and diagnostic bars invite an independence interpretation. | Five seeds reuse 93 videos; diagnostic counts reuse fitted models and masks. | Increase size of unit labels and retain mean-versus-seed distinction. | Bootstrap intervals remain fixed-fit conditional estimates. |
| Editor/designer | Major: 6.7-point labels were readable enlarged but weak at final paper size. | V1-V4 rendered pipeline and result panels. | V5 uses minimum 20.5 vector units at 1,000-unit width, about 8.1 pt at 5.5 inches; short labels and long final arrow segments. | Dense figure; inspect every rendered label for clipping. |

The reflection loss is deliberately absent from the main figure because it did not run in the completed gait grid. Its synthetic-only evidence remains in a short appendix paragraph.
