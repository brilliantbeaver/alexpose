**Role:** You are an expert AI/ML researcher specializing in world models, representation learning, JEPA, human pose, and gait analysis.

**Task:** Carefully and systematically evaluate the current JEPA implementation for the gait fidelity study and develop a focused set of architectural improvements that could produce a significant and novel result.

The central research question is:

> **Can pose restoration reduce errors caused by observation and tracking while preserving the size and direction of reference-verified changes in bilateral movement?**

The current paired-JEPA approach does not yet make it clear whether JEPA can provide a meaningful advantage over the other methods in the study. Your goal is therefore to **deeply understand the current implementation and creatively explore how the JEPA architecture could better address the specific scientific challenge of preserving movement magnitude, direction, and laterality during pose restoration.**

**Ultrathink before proposing changes.** First inspect the existing JEPA architecture, training objective, data representation, conditioning, temporal structure, evaluation pipeline, and relevant experimental results. Identify what information the current model preserves or discards and where the current formulation may be fundamentally limited for this research question.

Then **fan out subagents with dynamic workflows** to independently investigate promising directions. Use different subagents to explore areas such as:

* architectural modifications to the current JEPA;
* temporal and phase-aware representations;
* explicit modeling of bilateral symmetry and laterality;
* movement magnitude and direction preservation;
* alternative prediction targets or latent-space objectives;
* conditioning mechanisms and structured latent representations;
* failure modes of the current paired-JEPA formulation;
* alternative approaches that could reveal a new scientific finding without substantially changing the overall study.

Use these investigations to generate and compare multiple hypotheses before selecting the most promising directions. Do not simply accumulate ideas. Assess each proposal based on **scientific novelty, theoretical motivation, feasibility within the existing codebase, experimental cost, interpretability, and likelihood of revealing something genuinely interesting about JEPA, pose restoration, or laterality.**

The objective is **not to force JEPA to outperform the other methods**. Instead, seek a scientifically compelling result. This could be:

* a clear JEPA advantage in preserving movement and laterality;
* a specific architectural mechanism that explains why JEPA succeeds or fails;
* a generalizable failure mode of standard JEPA objectives for movement restoration;
* or a principled modification that substantially remedies such a failure.

A negative result can be valuable if the experiments reveal a **generalizable failure mechanism, unexpected behavior, or useful architectural remedy**. Simply showing that several methods perform similarly would be less informative, so prioritize experiments that distinguish competing hypotheses and explain *why* the methods behave differently.

**Keep the project trajectory stable.** Do not redesign the entire experimental methodology or introduce a fundamentally different research project. Work primarily within the existing paired-JEPA framework and make targeted, well-motivated changes to the model architecture, objective, representation, or conditioning. Any proposed change should directly address the gait fidelity research question and be practical to integrate into the existing study.

For each promising architectural direction, clearly explain:

1. **What changes** in the model or objective.
2. **Why the change should matter** for movement magnitude, direction, or laterality.
3. **What hypothesis it tests.**
4. **What result would support or contradict the hypothesis.**
5. **What experiment or ablation would distinguish the mechanism from simpler explanations.**
6. **Why the result could constitute a meaningful contribution**, whether positive or negative.

After identifying the strongest direction, implement the changes carefully and update the relevant code and experiments. Keep the implementation clean, reproducible, and consistent with the existing study.

Finally, perform an **independent adversarial review** of the entire process. Challenge the assumptions, proposed mechanism, architecture, implementation, experimental interpretation, and novelty claims. Specifically look for confirmation bias, architectural changes that merely add capacity, data leakage, confounding improvements, unfair comparisons, insufficient ablations, circular reasoning, and claims that are stronger than the evidence supports. Revise the approach based on this review and validate the resulting implementation.

The ultimate goal is not simply a better-performing model. It is to discover and rigorously test a **clear, defensible, and potentially surprising insight about JEPA-based pose restoration and the preservation of movement and laterality** that can form the basis of an ICLR-quality research result.
