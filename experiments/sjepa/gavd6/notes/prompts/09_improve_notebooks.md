**Role:** You are an expert AI/ML researcher specializing in world models, representation learning, and JEPA.

**Task:** Carefully and systematically revise the tutorial notebooks in `notebooks/gait_fidelity` so that they expose the key algorithmic and mathematical components of the gait fidelity study rather than hiding them behind high-level function calls.

The goal is for a technically strong reader to work through the notebooks and understand, at a deep level, how gait fidelity works, including the model architecture, representations, training procedure, prediction mechanism, losses, and evaluation metrics. The notebooks should demonstrate the underlying computations with clear, simple Python and machine-learning code wherever practical.

Use the notebooks in `/Users/theodoremui/dev/alexpose/experiments/multiple-sclerosis` as a reference for the desired level of transparency and clarity. Follow their approach of explicitly showing important model and mathematical operations while keeping the implementation readable and understandable.

**Ultrathink before making changes.** First inspect the relevant notebooks, source code, utilities, configurations, and supporting documentation to understand how the existing gait fidelity implementation works. Trace the complete flow from data and preprocessing through model architecture, training, inference, gait-fidelity computation, and evaluation before deciding what should be changed.

**Fan out subagents** to investigate the work in parallel where useful. Assign independent subagents to areas such as:

* understanding the existing gait fidelity implementation and experimental pipeline;
* comparing the target notebooks with the multiple-sclerosis notebooks;
* verifying the mathematical formulation and its correspondence to the code;
* identifying opportunities to expose hidden algorithmic steps;
* checking experimental methodology, metrics, and potential data leakage;
* reviewing the revised notebooks for clarity and correctness.

Use the findings from these investigations to inform the implementation, and reconcile conflicting conclusions yourself rather than blindly accepting individual subagent recommendations.

When revising the notebooks:

1. **Expose the algorithm.** Replace opaque calls with explicit implementations when doing so helps the reader understand the underlying algorithm. Clearly show the mathematical operations central to gait fidelity, including tensor transformations, representation construction, prediction, losses, and evaluation.
2. **Preserve correctness and behavior.** Maintain the existing experimental logic, outputs, data flow, and conclusions unless a change is necessary to correct an error or make the implementation more faithful to the intended algorithm.
3. **Prioritize clarity.** Introduce important concepts before implementing them, connect equations directly to the corresponding code, and use concise comments and explanations to clarify what the code computes and why.
4. **Avoid unnecessary reimplementation.** Do not expand every library or framework primitive into low-level code. Expose the scientifically important parts of gait fidelity while continuing to use standard libraries for routine operations.
5. **Make results interpretable.** Preserve or improve visualizations, metrics, intermediate outputs, and concrete examples so readers can verify what the algorithm is doing and understand the resulting behavior.
6. **Validate systematically.** Run the revised notebooks or the relevant code paths and verify that they execute correctly. Check tensor shapes, numerical behavior, reproducibility, data flow, train/evaluation separation, and consistency with the existing implementation. Compare important outputs against the original implementation where possible.
7. **Perform an adversarial review.** After implementing the changes, independently review the entire result as if trying to break it. Look specifically for incorrect mathematical translations, hidden assumptions, shape or indexing errors, accidental changes to the experimental methodology, data leakage, misleading explanations, unnecessary abstraction, and discrepancies between the equations, code, and stated algorithm. Fix every issue you identify and rerun the relevant checks.
8. **Review the complete tutorial flow.** Make sure the notebooks work together as a coherent sequence. A reader should be able to follow the progression from data and preprocessing through model architecture, training, gait-fidelity computation, and evaluation without needing to inspect hidden implementation details elsewhere in the repository.

Do not merely make superficial changes to comments or notebook text. The core implementation should genuinely expose the underlying computation where that improves understanding.

Before finishing, **Ultrathink and perform a final adversarial review** of both the implementation and the explanations. Ensure that the revised notebooks are scientifically faithful, executable, easy to follow, and substantially more transparent about how gait fidelity actually works, while preserving the study's intended outputs and experimental conclusions.
