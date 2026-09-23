**Role:** You are an expert AI/ML researcher specializing in world models, representation learning, and JEPA.

**Task:** Carefully and systematically revise the gait fidelity study across:

* `notebooks/gait_fidelity`
* `slurm/gait-fidelity`
* `src/gavd6_sjepa/research_directions/gait_fidelity`

The primary goal is to make the study capable of using the **full available AMASS and GAVD datasets** and to establish a robust, paper-ready experimental pipeline capable of producing meaningful and scientifically defensible results.

**Ultrathink before making changes.** First inspect the existing implementation end-to-end, including data loading and preprocessing, dataset splits, model configuration, training, evaluation, notebook workflows, and SLURM jobs. Determine how the current pipeline uses AMASS and GAVD, identify any limitations or unintended data restrictions, and understand the assumptions behind the existing experiments before modifying them.

**Fan out subagents with dynamic workflows** to investigate the problem in parallel where useful. Have independent subagents examine areas such as dataset usage and preprocessing, experimental design, model/training code, SLURM execution, evaluation methodology, and potential bugs or incompatibilities. Use their findings to guide the implementation and reconcile conflicting recommendations.

When making changes:

1. **Use the available data effectively.** Update the pipeline so that it can systematically use the full relevant AMASS and GAVD data rather than an unnecessarily restricted subset. Preserve scientifically appropriate subject-level or sequence-level separation where required to prevent leakage.
2. **Design paper-ready experiments.** Build a rigorous data-collection and evaluation pipeline that can support meaningful comparisons, sufficient sample sizes, reproducible experiments, and statistically interpretable results. Identify the experiments and measurements most likely to provide useful evidence for the gait fidelity research question.
3. **Preserve scientific validity.** Do not simply maximize dataset size. Carefully consider dataset heterogeneity, subject identity, sequence overlap, train/validation/test splits, conditioning variables, confounding factors, and whether the resulting measurements actually support the intended claims.
4. **Make the pipeline scalable.** Ensure the notebooks, source code, and SLURM scripts work together for large-scale data processing and training. Avoid approaches that are practical only for small exploratory subsets when the goal is full-dataset experimentation.
5. **Check the existing code thoroughly.** Look for bugs, incorrect assumptions, stale interfaces, incompatible APIs, incorrect paths or configurations, tensor-shape errors, data-loading problems, resource issues, and inconsistencies between notebooks, source code, and SLURM jobs. Fix issues that could prevent correct or reproducible execution.
6. **Preserve reproducibility.** Make dataset selection, preprocessing, splits, configurations, random seeds, experiment settings, and output locations explicit and reproducible. Avoid silently changing experimental conditions.
7. **Validate the complete workflow.** Run appropriate checks at each level, from data loading and preprocessing through training, evaluation, and SLURM execution. Where full-scale runs are too expensive, use smaller validation runs that exercise the same code paths before launching large experiments.
8. **Perform an independent adversarial review.** After implementing the changes, thoroughly review the result as if trying to find reasons the experiments could fail or produce misleading results. Look specifically for data leakage, invalid splits, duplicated samples, incorrect labels or conditioning, statistical weaknesses, implementation bugs, resource bottlenecks, incompatibilities, and discrepancies between the intended experiment and what the code actually executes. Fix discovered issues and rerun the relevant validation checks.

Do not make changes merely for completeness. Prioritize changes that materially improve the **scale, rigor, reliability, and scientific value** of the gait fidelity study.

Before finishing, perform a final **Ultrathink + adversarial review** of the entire pipeline and ensure that the notebooks, SLURM jobs, and source code form a coherent, executable system capable of collecting high-quality, paper-ready experimental results from the full available AMASS and GAVD data.
