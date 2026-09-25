**Role:** You are an expert AI/ML researcher specializing in world models, representation learning, and JEPA. You are also an exceptional technical communicator, scientific writer, and educator.

**Task:** Thoughtfully and systematically revise `docs/studies/gait-fidelity/proposal.html` and create a concise ~2-page version of the proposal that communicates the **significance, motivation, methodology, experimental progress, and scientific implications** of the gait fidelity study to a CS PhD student working across both **James Landay's Ambient Intelligence group at Stanford HAI** and **Scott Delp's biomechanics research**.

This task has **two related deliverables**:

1. **Revise the full proposal** to substantially improve its writing style, structure, motivation, and scientific narrative while preserving the useful technical depth of the existing document.
2. **Create a compressed ~2-page version** based on the revised proposal that communicates the most important ideas, evidence, and implications with much greater focus.

Do not treat the second document as a simple truncation of the first. First establish a strong, coherent research story in the full proposal, then distill that story into the shorter version.

### 1. Revise the full proposal

**Ultrathink about the existing proposal as a complete scientific argument before editing it.** Identify weaknesses in its current narrative, including overly procedural writing, vague motivation, unnecessary technical detail, weak transitions, and places where experiments are described without clearly explaining why they matter.

Reorganize and rewrite the longer proposal around a stronger writing style and structure:

**broader real-world problem → measurement challenge → gait fidelity problem → research question → methodological insight → experimental strategy → results → scientific implications.**

The revised proposal should feel like a **coherent research story rather than a sequence of experimental protocols or technical reports**.

Throughout the full proposal:

* Explain the reasoning behind major methodological and experimental decisions, not just what was done.
* Clearly distinguish the **problem**, **hypothesis**, **method**, **evidence**, and **implication**.
* Use standard terminology and conventional phrasing from AI/ML, biomechanics, pose estimation, and representation learning.
* Remove vague, generic, or unnatural language.
* Introduce technical concepts from first principles when needed, while preserving appropriate technical depth.
* Ensure each section naturally motivates the next.
* Avoid overstating results or significance.

### 2. Strengthen the scientific motivation

The motivation must clearly establish that gait fidelity addresses a broader problem of obtaining **reliable measurements of human movement from imperfect, noisy, and potentially incomplete observations**.

Explain concretely why this matters to both:

* **Biomechanics:** Reliable movement estimates are fundamental to measuring and interpreting human motion, comparing movement across conditions, and detecting meaningful changes in gait.
* **Ambient intelligence:** Real-world sensing systems must infer human movement from imperfect observations while remaining unobtrusive, robust, and useful outside controlled laboratory environments.

Make the connection to gait fidelity explicit. Explain why **pose restoration is not sufficient if the restoration changes the magnitude, direction, or laterality of the underlying movement**. Show why preserving these properties matters scientifically and why conventional pose-restoration approaches may not adequately guarantee them.

The proposal should make clear that this is fundamentally a **measurement and inference problem**, not merely an incremental improvement to pose estimation.

Do not force connections to the work of Scott Delp or James Landay, and do not make unsupported claims about either research program. Instead, explain naturally how reliable movement reconstruction could contribute to broader problems in biomechanics and ambient intelligence.

### 3. Create the focused ~2-page version

After revising the full proposal, create a substantially shorter version targeting approximately **2 pages**.

Use exactly these major sections:

1. **Introduction** — Establish the real-world problem and why reliable movement measurement matters to biomechanics and ambient intelligence.
2. **Methodology** — Explain the gait fidelity framework and key technical ideas clearly enough for a CS PhD audience.
3. **Experiments** — Explain what is being tested, why the experiments are necessary, and how they isolate the scientific question.
4. **Results** — Present the strongest current evidence and make the important findings immediately understandable.
5. **Discussion** — Interpret the results, explain their broader significance, acknowledge limitations, and connect the findings back to biomechanics and ambient intelligence.

The shorter version should preserve the **scientific logic and central narrative** of the full proposal rather than simply removing paragraphs. Prioritize the most important ideas, experiments, results, and implications.

### 4. Visual storytelling

Use graphics extensively, but ensure every graphic has a clear explanatory purpose. Prefer a small number of **high-information, research-quality visuals** over decorative figures.

For the **Introduction and Discussion**:

* Use carefully selected real-world images that make the movement-measurement, biomechanics, or ambient-intelligence problem intuitive.
* Use visuals to establish the real-world significance of the problem.

For the **Methodology, Experiments, and Results**:

* Create professional research-quality diagrams showing the model architecture, data flow, experimental design, and key findings.
* Use tables and graphs when they communicate quantitative results more effectively than prose.
* Make the relationship between the technical method and scientific question visually obvious.
* Maintain consistent notation, visual hierarchy, and terminology.

Every major figure should answer a clear question such as **“What is the problem?”, “How does the method address it?”, “How was it tested?”, or “What did we learn?”**

### 5. Adversarial review and parallel workflows

**Fan out subagents with dynamic workflows** to independently investigate and improve different aspects of the work, including:

* scientific motivation and significance;
* overall story and structure;
* technical accuracy;
* biomechanics and ambient-intelligence connections;
* experimental interpretation;
* visual design and figure quality;
* clarity and concision of the 2-page version.

Use their findings to improve the documents rather than simply accepting their conclusions.

Then perform an **independent adversarial review** of both versions. Actively look for:

* weak or unsupported motivation;
* unclear research questions;
* logical gaps or abrupt transitions;
* unnecessary technical detail;
* overclaiming;
* inaccurate descriptions of methods or results;
* weak connections to biomechanics or ambient intelligence;
* redundant content;
* graphics that are cluttered, ambiguous, or difficult to interpret;
* excessive text or overlapping visual elements;
* inconsistencies between the full proposal and the 2-page version.

Fix issues identified during the review and perform a final consistency check.

### Final standard

The **full proposal** should provide a technically rigorous and coherent account of the research while adopting a much stronger, more intuitive, significance-driven writing style.

The **2-page version** should function as a compelling research overview for a technically sophisticated reader who needs to quickly understand:

1. **Why reliable movement measurement from imperfect observations is important.**
2. **Why this problem matters to biomechanics and ambient intelligence.**
3. **What the gait fidelity study is asking that existing approaches do not adequately answer.**
4. **How the methodology addresses the question.**
5. **What the experiments have demonstrated.**
6. **What new scientific insight the work could contribute.**

The final result should be **clear, scientifically rigorous, visually compelling, and narratively coherent**, with the full proposal and 2-page version telling the same underlying research story at different levels of detail.
