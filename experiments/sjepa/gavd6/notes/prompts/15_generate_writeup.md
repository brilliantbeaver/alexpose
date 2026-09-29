**Role**: You are an expert AI/ML researcher specializing in world models, representation learning, and JEPA, with strong skills in scientific writing, technical education, and research visualization.

Write for a CS PhD student working across James Landay’s Ambient Intelligence group at Stanford HAI and Scott Delp’s biomechanics research, as well as an ICLR main-track reviewer. Assume strong general technical literacy without assuming familiarity with the study’s specialized terminology.

**Task**: Create two polished, evidence-grounded HTML drafts: a full scientific writeup based on `docs/studies/gait-fidelity/proposal.html`, and a focused companion of approximately two printed pages. Use the experiment outputs in `outputs/iclr`, existing documents, and relevant project code and documentation as your primary sources.

Develop the strongest paper claim that the evidence supports. Save clearly named new drafts alongside the existing documents, preserving the originals. Both versions should tell the same research story, supported by clear explanations and professional, publication-quality figures, diagrams, and tables.

**1. Establish the scientific argument and evidence**

Before editing, ultrathink about the study as a complete scientific argument. Identify its research question, hypothesis, methodological contribution, completed experiments, principal findings, and limitations. Examine where the current narrative is procedural, unclear, unsupported, or disconnected from the scientific question.

Verify descriptions of datasets, preprocessing, models, training, evaluation, and results against the available artifacts. Keep quantitative claims traceable to source files or tables, and cite external scientific claims appropriately.

Distinguish observed findings, interpretations, hypotheses, and proposed experiments. Identify missing information explicitly rather than inventing details or presenting planned work as completed.

**2. Explain technical concepts from first principles**

At first use, explain important specialized terms in plain language: what they represent, why they are needed, and how they work in this study. Expand acronyms, but do not treat expansion alone as an explanation.

Build from concrete observations to abstract ideas. Show what a movement sample contains before explaining its representation. Explain magnitude, direction, and laterality through specific movement examples before introducing formal measures.

For each central method or metric, clarify its inputs, operation, outputs, and scientific meaning. Use equations when they add precision, define their symbols, and explain how to interpret their values. Explain concepts such as latent representations, self-supervised learning, or JEPA only where relevant to the actual study.

Keep explanations concise and accurate. Pair difficult concepts with useful examples or visual aids without turning the writeup into a general textbook.

**3. Develop the full scientific writeup**

Organize the narrative around this progression:

Real-world problem → measurement challenge → gait fidelity question → methodological insight → experimental strategy → results → scientific implications.

Frame gait fidelity as a problem of reliably measuring human movement from noisy, imperfect, or incomplete observations. Explain how a plausible pose reconstruction can nevertheless alter the magnitude, direction, or laterality of the movement being measured, and how this study evaluates those changes.

Connect the problem naturally to biomechanics, where reliable measurements support comparisons and detection of meaningful gait changes, and to ambient intelligence, where unobtrusive sensing must work beyond controlled laboratory conditions. Avoid unsupported claims about either named research group or existing methods.

Explain what was done and why each major decision helps answer the research question. Describe data provenance, distinguish newly collected data from reused datasets, and explain the models actually employed. Include relevant training objectives, evaluation measures, baselines, controls, ablations, and uncertainty where available.

Preserve useful technical depth while moving procedural detail out of the main narrative when it interrupts the argument. Interpret negative and inconclusive results when they affect the central claim.

**4. Create the approximately two-page companion**

Write the short version after establishing the full research story. Recompose it around the most important ideas and evidence rather than truncating the longer document. Preserve concise explanations of essential technical concepts.

Use exactly these seven major sections:

1. Introduction — Establish the movement-measurement problem, its significance, and the research question.
2. Data collection — Explain data provenance, what a sample contains, and how it was used. Include representative examples from the actual study data.
3. Methodology — Explain the gait fidelity framework and its central concepts through a concrete example.
4. AI models and techniques — Identify the models, their roles, and essential training and evaluation details.
5. Experiments — Explain what each central comparison tests and how the design isolates the scientific question.
6. Results — Present the strongest findings with sufficient quantitative context to assess them.
7. Discussion — Interpret the evidence, acknowledge limitations, and explain implications for biomechanics and ambient intelligence.

Treat two pages as a rendered print-layout target, including figures and references. Achieve concision through prioritization, not smaller text or crowded graphics.

**5. Use concrete, explanatory visual aids**

Choose visuals because they improve understanding of a specific topic. Every major visual should answer a clear question about the problem, method, experiment, or finding.

In Data collection, show representative samples from the data actually used: image frames, pose sequences, joint trajectories, or sensor traces, depending on the available modalities. Explain what the reader is seeing and how it enters the analysis. Label relevant joints, timestamps, coordinate systems, units, and conditions. Generic imagery of someone walking does not substitute for showing the study data.

Where the artifacts permit, follow the same example through the document: original observation, corruption or missing information, model input, reconstruction, and fidelity evaluation. Preserve sample identity and comparable scales.

Use annotated diagrams to explain representations, architecture, data flow, and learning objectives. Use empirical plots and tables to show how the results answer the research question. Include contextual real-world imagery only when it adds explanatory value.

Identify sample provenance and distinguish illustrative diagrams from experimental evidence. If an actual example is unavailable, state the limitation.

**6. Meet professional research publication standards**

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

**7. Coordinate review and verify the final artifacts**

Use parallel workflows: fan out subagents for bounded, independent reviews of scientific motivation, narrative structure, technical accuracy, experimental interpretation, first-principles explanations, visual quality, and concision. Allocate work according to the issues discovered, and reconcile recommendations against the evidence.

After integrating revisions, obtain an independent adversarial review of both drafts. Challenge unsupported claims, unexplained terminology, methodological inaccuracies, confounded comparisons, logical gaps, weak implications, and inconsistencies between versions. Inspect every major visual for scientific accuracy, explanatory value, legibility, and misleading presentation.

Fix identified issues, then render and inspect both HTML documents and their print layouts at the intended reading size. Check figure values against source outputs, caption consistency, table alignment, page breaks, clipping, overlap, and broken assets or links.

Deliver both drafts and briefly report the central claim, principal improvements, and remaining evidence gaps.
