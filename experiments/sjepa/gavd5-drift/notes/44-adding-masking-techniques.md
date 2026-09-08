
In the notebooks and documents in "neurips-laterality", we have relied exclusively on a fixed number of (supposedly neurologically inspired) keypoints for masking, and for the encoding to learn.  Ultrathink if this is too restrictive.  Deeply research from authoritiative sources such as ArXiv on what are other popular and promising masking techniques that we should have experimented with in comparison?  

Write your response using simple, fluent, grounded, and easy to understand language in the file: neurips-laterality/docs/TUTORIAL.md -- using lots of examples, specific illustrations to clarify any ideas or concepts, and with well structured arguments and writing.

Fan out subagents to parallelize your planning and task executions.

---

In your writeup, craft an appropriate instructional prompt to properly setup the notebooks that are needed to expand the comparative masking techniques, and their resulting encoder training, and predictor performance comparisons.  Be as systematic and meticulous as possible in your instruction writing.

---

Carefully and systematically update the new notebooks 11, 12, 13, 14 in "neurips-laterality" with the same "notebook\_progress.py" setup as the other earlier notebooks to track progress on long running cells.

---

In notebook 12, we don't need the progress tracking for the cell at the top.  Instead, let's reuse how we optimize running of training from the earlier notebooks by maximizing MLX / Mac GPU / Nvidia GPU and parallelism.  With the current approach, the following cell would have taken too long to execute: ultrathink on how to optimize the loop & runtime performance based on as much hardware acceleration and caching as possible:

...

---

Deeply review the "neurips-laterality" notebooks: 11, 12, 13, 14, summarize for me what has been learned from running of these experiments, particularly about masking.  Update the document neurips-laterality/docs/TUTORIAL.md with our findings with lots of examples, and any appropriate vector graphics. 

Also carefully update the executed notebooks with commentaries & interpretations on the executed results in the notebooks.

Fan out subagents to parallelize your planning and task executions.
