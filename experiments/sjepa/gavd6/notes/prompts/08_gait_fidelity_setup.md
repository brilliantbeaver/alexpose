**Role**: You are an expert AI/ML researcher specializing in world models and JEPA.

**Task**: You are to carefully and systematically set-up and build the experiments for the full Gait Fidelity study, described in detail in `docs/studies/gait-fidelity`.

Ultrathink on how to implement each experiment as a set of Jupyter notebook tutorials so that the user can follow along each step of the run. Use the following directories for each aspect of your implementation:

- source code: src/gavd6\_sjepa/research\_directions/gait-fidelity
- slurm scripts: slurm/gait-fidelity
  - based on the README.md files for other studies within the slurm folder, thoughtfully create a systematic, clear, and intuitive guide to executing the experiments on my HAIC environment. Make sure to thoroughly check past files and your conversation history for the latest status on my environment.
- notebooks: notebooks/gait_fidelity
- experiment writeup: `docs/studies/gait-fidelity`. Use the same writing style as the proposal to clearly communicate the experiment setup and carefully explain what each experiment does.

Your writing should be natural, fluent, grounded, and easy to understand and to follow. Avoid common LLM styling and characteristics in your response. Fully explain any technical jargon in clear, simple terms.

Use independent adversarial review to thoroughly check your work and make sure that there are no errors. The experiments must be immediately runnable on HAIC, with no bugs, missing dependencies, incompatibilities, etc. Refernece previous synthetic training studies for examples of what works.

Use fan out subagents with dynamic workflows.
