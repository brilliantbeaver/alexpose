**Role**: You are an expert world model (JEPA) researcher well versed in human gait analysis with world models

**Task**: Ultrathink how best to modify and add to the current notebooks in the following "Active Folder" to reify idea 9 on how to separate lateralized from symmetric gait by construction of the S-JEPA encoder:

Active Folder: "experiments/sjepa/gavd6-pm"

The idea 9 description, methodology and context can be found in this folder: experiments/sjepa/gavd5-drift/notes/ideas-claude/09-reflection-equivariant-symmetry-axis.

Output your plan and implementation in a well structured markdown file in the Active Folder systematically and thoroughly.

## Research Idea 09 Context

In the "Active Folder", we have already implemented a set of notebooks that have proven using Idea 05 on Signed laterality decodability that the trained frozen encoder does not expose the tested signed laterality axis strongly enough to pass the three gates.

* The learned S-JEPA curriculum encoder did not organize a directional asymmetry axis at all.
* We cannot decode a signed left-minus-right asymmetry axis linearly decodable from the frozen tokens above a raw-coordinate null
* An anatomical mirror does not flip its sign in its representation.

Both Idea 05 and Idea 09 and their methodologies can be found in subfolders of: experiments/sjepa/gavd6-pm/notes/ideas-claude

## Output Instructions

In the "Active Folder", clearly document the Idea 9 experimental methodologies, datasets, and expected results for different versions of possible futures in both README.md as well as a well-versed well structured Markdown document with well illustrated descriptions, logic, reasoning, and scientific rigor.  Use natural and simple languages in your narrative and logical flow.  Illustrate with plenty of specific examples, easy-to-follow tutorials, as well as clean and non-cluttered vector graphics.  Your audience is an advanced high school student with little background in this area.

Use codex:adversarial-review to double-check your plan, logic and implementation.  Make sure that your output is correct and methodologically sound.  You should only make statements or conclusions that can be verified.

Fan out subagents with dynamic workflows.
