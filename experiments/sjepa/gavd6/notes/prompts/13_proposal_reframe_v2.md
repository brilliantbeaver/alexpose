**Role:** You are an expert AI/ML researcher specializing in world models, representation learning, and JEPA. You are also an exceptional technical communicator, scientific writer, and educator.

**Task:** Thoughtfully and systematically revise `docs/studies/gait-fidelity/proposal.html` so that the proposal is substantially more intuitive, logical, precise, and compelling.

First, **Ultrathink about the proposal as a whole** rather than editing sections independently. Understand the research question, motivation, methodology, and intended contribution before rewriting. Identify weaknesses in the current story, including vague or unnatural language, unclear reasoning, unnecessary technical detail, abrupt transitions, and places where experimental procedures are presented without explaining why they matter.

### Writing and scientific communication

Rewrite each section from **first principles** so that a technically sophisticated reader can easily understand:

* What problem the study addresses and why it matters.
* What makes the problem difficult.
* What the central research question is.
* Why the proposed approach is appropriate.
* Why each major experimental decision is necessary.
* What each experiment is intended to reveal.
* How the results could contribute to a broader scientific problem.

The proposal should read as a **coherent scientific argument**, not a sequence of experimental protocols or technical reports. Prioritize the reasoning behind the experiments and the scientific insight they are designed to produce.

For technical AI/ML concepts, use **standard terminology and conventional phrasing** from the relevant literature. Do not invent terminology, unnecessarily rename established concepts, or use unusual wording when a commonly accepted term exists. Preserve technical precision while explaining concepts in an intuitive way.

### Broader research connection

Develop a clear connection between gait fidelity and one or both of the following research areas:

* **Professor Scott Delp's biomechanics research at Stanford**, particularly the use of computational methods and human movement analysis to understand biomechanics.
* **Professor James Landay's Ambient Intelligence work at Stanford HAI**, particularly the use of intelligent, unobtrusive sensing to understand human behavior and movement in real-world environments.

Do not force these connections or make unsupported claims about either research program. Instead, explain naturally how the gait fidelity problem fits into the broader challenge of obtaining **reliable measurements of human movement from imperfect real-world observations**, and why solving that problem could matter for biomechanics, ambient intelligence, or both.

### Story and structure

Deeply analyze the proposal's overall story arc and revise it so that each section naturally motivates the next. The reader should be able to follow a clear progression:

**broader problem → measurement challenge → gait fidelity problem → research question → proposed approach → experimental design → expected scientific insight → broader significance.**

Every major experiment should have a clear purpose in the narrative. Explain not only **what will be measured**, but **why that measurement matters and what knowledge it could provide**.

Avoid overstating the potential impact. Clearly distinguish established facts, hypotheses, expected findings, and possible implications.

### Mathematical presentation

Carefully inspect every mathematical equation and improve its LaTeX rendering where necessary. Equations should display correctly in the HTML and use clean, conventional notation.

**Do not use the `\operatorname` macro.** Use simpler standard LaTeX notation that renders reliably in the proposal.

Check that mathematical notation is consistent throughout the document and that every equation is explained clearly enough for the reader to understand what it represents and why it is relevant.

### Review and validation

**Fan out subagents with dynamic workflows** to independently review different aspects of the proposal, such as:

* scientific logic and research motivation;
* technical accuracy and AI/ML terminology;
* story arc and section transitions;
* mathematical notation and LaTeX rendering;
* connections to biomechanics and ambient intelligence;
* clarity, naturalness, and readability.

Then perform an **independent adversarial review** of the revised proposal. Actively look for vague claims, unsupported conclusions, unnecessary jargon, invented terminology, logical gaps, weak transitions, disconnected experiments, overclaiming, mathematical inconsistencies, and places where the motivation does not actually justify the proposed experiment.

Use the review findings to make a final revision. Do not merely polish the wording; make substantive changes wherever they improve the scientific reasoning or narrative.

The final proposal should feel like **one coherent research story**: a clear and important real-world problem motivates a precise research question, the research question motivates the experiments, and the experiments are designed to produce interpretable scientific insight that connects back to the broader problem.

---

**Role:** You are an expert AI/ML researcher specializing in world models, representation learning, and JEPA. You are also an exceptional technical communicator, scientific writer, and educator.

**Task**: Thoughtfully and systematically create a compressed, focused version of the document in `docs/studies/gait-fidelity/proposal.html` to convey the significance and progress of the gait fidelity study to a CS PhD student working in both James Landay's ambient intelligence lab as well as Scott Delp's biomechanics lab.

You MUST provide a much much stronger motivation section for how the problem that the gait fidelity study is trying to address is related and helpful to the ambient intelligence and biomechanics work in a meaningful way. Make sure to clearly explain how this research is impactful and significant, rather than simply an incremental improvement.

Ultrathink on how to make the storytelling much more powerful and compelling. The entire writeup should be shorter than the current proposal, around 2 pages in length. You should include an introduction, methodology, experiments, results, and discussion section. Make sure to illustrate frequently and abundantly with clear and insightful images, tables, graphs, etc. For the introduction and discussion, use real world images that illustrate the real world problem. For the methodology, experiments, and results, create highly professional and research quality diagrams and highlight the scientific and technical rigor of the gait fidelity study. 

Use independent adversarial review of your graphics and workflow to ensure that they are not cluttered, and that no excessive text or line overlaps. Each generated image should be easy to understand and with minimal clutter.

Use fan out subagents with dynamic workflows to parallize your tasks.

---


**Role:** You are an expert AI/ML researcher specializing in world models, representation learning, and JEPA. You are also an exceptional technical communicator, scientific writer, and educator.

**Task:** Thoughtfully and systematically create a concise, focused version of `docs/studies/gait-fidelity/proposal.html` that communicates the **significance, motivation, methodology, experimental progress, and scientific implications** of the gait fidelity study to a CS PhD student working across both **James Landay's Ambient Intelligence group at Stanford HAI** and **Scott Delp's biomechanics research**.

The new document should be substantially shorter and more focused than the current proposal, targeting approximately **2 pages**. Do not simply shorten the existing text. Reconstruct the narrative around the most important scientific ideas and evidence.

### Core objective: strengthen the motivation

The motivation must be substantially stronger than in the current proposal.

Clearly explain the broader problem of obtaining **reliable measurements of human movement from imperfect, noisy, and potentially incomplete observations**. Then show why this problem matters to both:

* **Biomechanics:** Reliable estimates of movement are fundamental to measuring and interpreting human motion, comparing movement across conditions, and understanding meaningful changes in gait.
* **Ambient intelligence:** Real-world sensing systems must infer human movement from imperfect observations while remaining unobtrusive, robust, and useful outside controlled laboratory environments.

Make the connection to the gait fidelity study explicit and concrete. Explain why **restoring or correcting observed pose is not sufficient if the restoration changes the magnitude, direction, or laterality of the underlying movement**. Establish why preserving these properties is important scientifically and why existing pose-restoration approaches may not adequately guarantee it.

The motivation should make clear that gait fidelity addresses a **broader measurement and inference problem**, rather than presenting the work as a small incremental improvement to pose estimation. Explain what becomes possible if movement can be restored while preserving scientifically meaningful changes, and what kinds of biomechanics or ambient-intelligence systems could benefit from such a capability.

Do not exaggerate the significance or make unsupported claims. The argument should be compelling because the underlying problem and potential insight are important, not because of inflated language.

### Storytelling and structure

**Ultrathink about the story before writing.** Identify the smallest set of ideas needed to make the research compelling and organize them into a clear progression:

**real-world problem → limitations of current observations → gait fidelity challenge → research question → proposed solution → experimental evidence → scientific insight → broader implications.**

Use exactly these major sections:

1. **Introduction** — Establish the real-world problem and explain why reliable movement measurement matters to biomechanics and ambient intelligence.
2. **Methodology** — Explain the gait fidelity framework and the key technical ideas clearly enough for a CS PhD audience to understand the approach.
3. **Experiments** — Explain what is being tested, why each experiment is necessary, and how the experimental design isolates the scientific question.
4. **Results** — Present the strongest current evidence and make the important findings immediately understandable.
5. **Discussion** — Interpret what the results mean, explain their broader significance, acknowledge limitations, and connect the findings back to biomechanics and ambient intelligence.

Prioritize **why** each methodological and experimental choice matters over exhaustive implementation details. The reader should understand the scientific logic without needing the full technical proposal.

### Graphics and visual storytelling

Use graphics extensively, but make every graphic serve a clear explanatory purpose. Prefer **a small number of high-information, research-quality visuals** over many decorative figures.

For the **Introduction and Discussion**:

* Use carefully selected real-world images that make the sensing, movement-measurement, biomechanics, or ambient-intelligence problem immediately intuitive.
* Images should provide context that is difficult to communicate through text alone.

For the **Methodology, Experiments, and Results**:

* Create professional research-quality diagrams that clearly communicate the model architecture, data flow, experimental design, and key findings.
* Use tables and graphs where they communicate quantitative results more effectively than prose.
* Make the relationship between the technical method and the scientific question visually obvious.
* Prefer clean diagrams with strong visual hierarchy, minimal text, and consistent notation.

Every figure should answer a specific question such as **“What is the problem?”, “How does the method address it?”, “How was it tested?”, or “What did we learn?”**

### Visual quality control

**Fan out subagents with dynamic workflows** to work in parallel on the writing, scientific narrative, methodology, experimental interpretation, and graphics.

Use **independent adversarial review** to evaluate both the document and every major graphic before finalizing. Specifically check for:

* clutter and excessive visual elements;
* text that is too small or unnecessarily dense;
* overlapping lines, labels, or objects;
* ambiguous arrows or data flow;
* inconsistent notation;
* misleading visual encodings;
* graphics that require excessive explanation;
* redundant figures;
* unsupported scientific claims;
* weak connections between figures and the central research question.

Revise or regenerate graphics that fail these checks.

### Final standard

The final document should feel like a **concise research pitch to a technically sophisticated scientist**, not a shortened technical report.

A reader should finish the ~2-page document understanding:

1. **Why reliable movement measurement from imperfect observations is an important problem.**
2. **Why that problem matters specifically to biomechanics and ambient intelligence.**
3. **What the gait fidelity study is asking that existing approaches do not adequately answer.**
4. **How the proposed methodology addresses that question.**
5. **What the experiments have demonstrated so far.**
6. **What new scientific insight the work could contribute.**

Optimize for **clarity, scientific significance, narrative coherence, and visual communication** while preserving technical rigor and accurately representing the current state of the research.
