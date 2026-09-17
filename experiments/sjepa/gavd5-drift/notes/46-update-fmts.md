**Role**: You are an expert in AI/ML, specializing in JEPA and gait analysis.

**Task**: You are to first systematically review the Skeleton JEPA implementation & results in the `neurips-laterality` folder (and subfolders) to understand weaknesses and shortcomings of the current approach for the NeurIPS workshop on Foundation Models for Temporal Systems: https://fmts-workshop.github.io/.

Deeply reflect on our progress in these notebooks so far, ultrathink suggest changes to the setup and configuration so as to significantly improve the quality and significance of the results. You should plan by first understanding latest literature about using Self-Supervised Learning on video gait analysis, as well as video-based JEPA from authoritative sources. Be clear on what directions you would like to explore for this goal by documenting your plan as "notes/47-improvement-plan.md" with specific details and suggestions.

For the revised experiments, use the full GAVD videos and walking sequences, which are stored on my HAIC environment. Do not attempt to search for these sources, as they are in a separate location. Use the same format and structure as the source code in `gavd6/src/gavd6_sjepa/research_directions/motion_preservation` and the Slurm scripts in `gavd6/slurm/motion-preservation`. Output detailed and specific documentation for the user to easily follow and run the generated experiments.

Based on your recommended suggestions and next steps, thoughtfully and deeply create a detailed instruction to a frontier thinking LLM such as GPT-6 Astra and Fable 5.1 to correct and refine the notebooks here so that we can significantly improve the results of the current S-JEPA approach. Include instructions to use Codex for adversarial-review, and for Claude Code to fan out subagents with dynamic workflows.  Use best practice prompting techniques to ultrathink on how to deeply research latest JEPA techniques using authoritative research such as those form ArXiv, ACM and IEEE sources.

Fan out subagents with ultracode and dynamic workflows.
