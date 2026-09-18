# Executive summary review — 13 September 2026

Reviewed document: [docs/09-0913-SUMMARY.md](../../docs/09-0913-SUMMARY.md).

The independent review approved factual accuracy and inference. Its three clarity suggestions were addressed: the window count now describes the collection without implying one training partition; the additional 400 updates explicitly restart from the saved model; and the conclusion states directly that a JEPA advantage has not been established. Additional edits explain technical terms in plain language.

The four pooled scores and the S-JEPA-minus-Random-Forest difference were checked against `artifacts/runs/r1_g1_1k_s42/results.json` and `artifacts/eval/g1/E0_results.json`. Both links in the summary resolve. The notebook demonstration scores were cross-checked against the detailed report and its retained-notebook evidence audit.

The installed `codex:adversarial-review` command ran with `--wait --scope working-tree`, focused on the summary and its supporting evidence. Its final output was:

> # Codex Adversarial Review
>
> Target: working tree diff
> Verdict: approve
>
> Approve: claims match the detailed report and retained evidence. The summary separates the pre-centering five-fold result from current single-fold demonstrations and avoids unsupported claims about statistical reliability, participant independence, or symmetry mechanisms.
>
> No material findings.

## Writing-style refinement and training-step clarification

The user requested a clearer motivation, an explicit working null, a connected notebook sequence, illustrations, and another adversarial review while preserving the summary's length. Two parallel reviewers checked the scientific boundaries and narrative/figure choices. The revised summary uses the existing symmetry-cycle and protocol-separation SVGs.

The review suggestions were addressed as follows:

- Clinical motivation now distinguishes Parkinson’s arm-swing asymmetry from MS inter-joint coordination, with both primary study abstracts verified. Neither association is presented as established by this collection.
- The working null is explicit and retrospective. Its status is limited to what the retained notebooks document, without asserting an undocumented history of hypothesis formation.
- Normalization specifies both spatial coordinates, with visibility unchanged. Tokenization preserves temporal order within each joint; the augmentation description matches the shared transformations and left/right relabeling in `sjepa/augment.py`. Learned geometry preservation remains unproven.
- Proposed group comparisons and individual-joint comparisons are distinguished, avoiding definitive attribution from a bundled feature result.
- The main outcome appears near the introduction. Captions identify synthetic curves and explain the evaluation terminology.
- In response to the user's question, 800 updates are defined as 800 training steps using repeatedly sampled batches of 32 windows. This was verified against the retained notebook configuration and training sampler. Test sources remain excluded.

The installed `codex:adversarial-review` command was run again with `--wait --scope working-tree`, focused on the revised summary and its two SVGs. Its final output was:

> # Codex Adversarial Review
>
> Target: working tree diff
> Verdict: approve
>
> No material blockers found in the summary or its two SVGs. Retained scores and methods support the claims; prose and captions distinguish motivation from results, limit shape-preservation claims, disclose selected-best controls, and separate the older five-fold reference from current single-fold demonstrations. Both figures render clearly.
>
> No material findings.

Subsequent wording refinements, including the training-step explanation, were independently checked against the code. The final summary's numerical claims and relative links were checked, and an HTML preview was rendered in Chrome for visual inspection. No notebook, training output, or shared SVG was changed.
