# Review of revision 2 → decisions for revision 3

Revision 2 makes the training and splitting procedure substantially easier to audit. Its main weakness is that it gives a range of trained scores without explaining the individual comparisons, and still postpones the input-measurement problem.

1. Replace the teacher range with all five trained arms, seed variation, MAE and matched controls. Identify repeated baselines and distinguish seed SD from source intervals.
2. Add the independently repeated Notebook 07 calculation with its exact overlap cohort. State that agreement after preparation is not model accuracy or a maximum attainable R².
3. Add a separate, clearly post-hoc bootstrap of trained-minus-initial scores. The original mask contrasts cannot supply uncertainty for a different estimand.
4. Explain that the initial S-JEPA baseline retains a pretrained pose detector and anatomy-informed supervised readout.
5. Correct the student description: hidden coordinate content is suppressed within the token grid; masked positions and positional embeddings remain.
6. Explain why the mask interventions are real: target-motion enrichment and lost temporal brackets are observed. Keep different mask families paired with their own budgets.
7. Include online-encoder evidence and the boundary-selected ridge penalties. Neither teacher selection nor modest penalty tuning resolves the learning deficit.
8. Avoid treating numerical changes between Notebooks 08, 12 and 18 as causal effects of masking or regularization.

Revision 3 implements these requests. It leaves the original rounded reflection results as historical context rather than relabeling them independently reproduced evidence. Independent audits of the source code and retained predictions contributed to these changes.

Reviewer score for v2: 70.3/100. Remaining work concerns literature, medical motivation, competing explanations, and editing the growing material into a focused workshop narrative.

