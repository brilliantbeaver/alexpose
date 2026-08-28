**Role**: You are an expert in AI/ML, specializing in JEPA and gait analysis.

**Task**: You are to first systematically review the Skeleton JEPA implementation & results in this folder (and subfolders) to understand weaknesses and shortcomings of the current approach.

Deeply reflect on our progress in these notebooks so far, ultrathink suggest changes to the setup and configuration so as to significantly improve the classification accuracy of the S-JEPA approach in 06_capstone_health_condition_classifiers.ipynb.  You should plan by first understanding latest literature about using Self-Supervised Learning on video gait analysis, as well as video-based JEPA from authoritative sources.  Be clear on what directions you would like to explore for this goal by documenting your plan as "notes/04_improvement_plan.md" with specific details and suggestions.

Based on your recommended suggestions and next steps, thoughtfully and deeply create a detailed instruction to a frontier thinking LLM such as Anthropic Opus 4.8 and Fable 5 to correct and refine the notebooks here so that we can significantly improve the classification accuracy of the S-JEPA approach.  Include instructions to use Codex for adversarial-review, and for Claude Code to fan out subagents with dynamic workflows.  Use best practice prompting techniques to ultrathink on how to deeply research latest JEPA techniques using authoritative research such as those form ArXiv, ACM and IEEE sources.

For each of the new "normal" video in data-videos/normal, use your best video pose estimation capabilities (such as MediaPipe) to first annotate individual gait sequences with bounding box on the moving subject.  Use these results & information to write to the GAVD "csv" file for processing by the notebooks.

To output your prompting instructions only, write to "notes/05_improvement_instr.md"

Fan out subagents with ultracode and dynamic workflow
