**Role**: You are an expert AI/ML researcher specializing in world models, representation learning, and JEPA, with strong skills in paper writing, scientific reasoning, and research visualization.

**Task**: First, deeply read & understand the work that is describe in the documents in the `docs/studies/gait-fidelity` folder.

Then, fully understand the experiment outputs for the gait fidelity study in `outputs/gait-fidelity`, as well as the experiment notebooks in `notebooks/gait_fidelity`.

Ultrathink on what are possible ways to frame this research on pose fidelity in movement restoration to successfully write a 9 page paper excluding references focusing on the laterality approach and results for the International Conference on Learning Representations (ICLR):

* https://iclr.cc/Conferences/2027/CallForPapers

You must then systematically and thoughtfully develop the strongest paper claim and writeup that the evidence supports. Critically and creatively think about how to generate your writeup as a complete scientific argument. Identify its research question, hypothesis, methodological contribution, completed experiments, principal findings, and limitations. Examine where the current narrative is procedural, unclear, unsupported, or disconnected from the scientific question. The writeup should be compelling, scientifically relevant, and very exciting to readers and reviewers. Make sure to only use evidence that is available in the results, notebooks, or existing documents, and do not make up claims whatsoever.

Iteratively create 7 versions of the paper in the folder `docs/iclr` without overriding existing files by thoughtfully incorporating each of the successive versions' critique and suggestions. Each version must improve upon previous versions by carefully reviewing and selecting the most appropriate results, inferences, and findings, as well as relevant literature from authoritative primary sources such as research papers and their official implementations.

Use the same scoring rubric throughout: relevance and contribution to the conference (20%), claim accuracy and evidence support (20%), evaluation and statistical rigor (15%), scientific insight and positioning against related work (15%), reproducibility (10%), clarity and narrative (10%), figures (5%), and submission fit (5%). Score each dimension from 0 to 10 and report the weighted total out of 100, with concrete reasons and remaining weaknesses. A score of 5 indicates a substantial unresolved weakness and 10 indicates no material weakness within the stated scope. Distinguish improvements achievable through revision from limitations that require new data or experiments; a later version should not receive a higher score merely because it is later.

Use concrete, explanatory visual aids throughout your writeup. Choose visuals because they improve understanding of a specific topic. Every major visual should answer a clear question about the problem, method, experiment, or finding.

Use annotated diagrams to explain representations, architecture, data flow, and learning objectives. Use empirical plots and tables to show how the results answer the research question. Include contextual real-world imagery only when it adds explanatory value.

Treat figures, diagrams, and tables as scientific arguments that must withstand close reviewer scrutiny.

For all visuals:

- Use consistent typography, notation, colors, spacing, and panel labels across both documents. Ensure labels remain readable at their final displayed and printed sizes.
- Prefer vector graphics for diagrams and plots, and sufficiently detailed raster images for actual data examples. Retain editable sources where practical.
- Use an accessible, restrained palette. Combine color with labels, symbols, or line styles so meaning survives grayscale reproduction.
- Write self-contained captions explaining what is shown, how to read it, the experimental context, and the principal takeaway.
- Remove decoration, redundant labels, unnecessary borders, and other elements that compete with the evidence.

For diagrams, make relationships and arrow meanings explicit. Distinguish training from inference and observed data from predictions where relevant. Match the actual implementation and avoid ambiguous boxes or unexplained symbols.

For quantitative plots, label axes and units, identify conditions and baselines, and use comparable scales for comparisons. Explain aggregation, sample sizes, and uncertainty where available; define error bars precisely. Do not manufacture uncertainty or use graphical choices that exaggerate effects.

For tables, use descriptive headers, units, consistent precision, and clear metric direction where needed. Align numeric columns, distinguish unavailable values from zeros, and explain abbreviations. Highlight key findings sparingly and only when the comparison supports doing so.

Keep the title "Evaluating Feature Prediction for 2D Pose Trajectory Restoration with Paired Synthetic Supervision".

## Writing Style

Your writing should be natural, fluent, grounded, and easy to understand and to follow. Avoid common LLM styling and characteristics in your response. Fully explain any technical jargon in clear, simple terms.

Use independent adversarial review to carefully and thoughtfully review your writeup and propose suggested changes. For each version, examine the manuscript from the perspective of a skeptical world models researcher, an evidence auditor, a statistical reviewer, and an editor enforcing the paper scope and natural writing. Record each substantive objection, its severity, the evidence, the correction or reasoned disposition, and any residual limitation in the ICLR review record. Based on these suggestions, systematically revise the paper and address all feedback. A final pass must find no unresolved material misstatement; weaknesses requiring new experiments should remain plainly bounded rather than rewritten as wolved.

## Avoidance

Your output must avoid common LLM output styling and characteristics:

* Staccato drumbeat sentences: short sentences
* Humans land an aphorism occasionally; LLMs land one every time, and they close nearly every section and the abstract this way.
* The "it is not X, it is Y" correction reflex that is highly correlated with LLM outputs.  This antithesis pattern appears throughout at high density.
* Recycled pivot phrases. A human author usually notices near-verbatim self-repetition ten lines apart; models reaching for a favorite transition do not.
    * "Confidence tells the same story from a different angle"
    * "The concurrency tier tells the same story from a slightly different angle"
    * "What looks like an architecture effect is noise"
* Intensifier tics. "Actually" appears 20 times, "exactly" 8 times: That is a lot of emphasis with no technical context.
* Anthropomorphic phrasing throughout.
* Groomed triads are dotted throughout: examples:
    * "Real, sharply structured, and immune to the standard fixes."
    * "Never the oracle, never the operator, and never which arm of the pair."
* The suspicious absences. Zero em-dashes at all, and zero instances of the classic AI lexicon (delve, leverage, robust, comprehensive, landscape, underscore). Most human ML writers use a dash or the word "robust" at least once.

Use fan out subagents with dynamic workflows.
