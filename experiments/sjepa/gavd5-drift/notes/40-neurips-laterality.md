**Role**: You are an expert AI/ML researcher specializing in world models, JEPA, and human gait analysis.

**Tasks**: First, deeply read & understand the work that are described in the documents in the "docs" folder.

Then, fully understand the jupyter notebooks (*.ipynb) and results in them.  Then ultrathink on what are possible directions for this S-JEPA research to successfully write an 8 pages paper, as well as a 4 pages extended abstract focusing on the laterality approach and results for this workshop on Physical World AI: Geometry, Characteristics, and Multimodal Sensing:

* https://physworld-org.github.io/physworld.github.io/cfp/

You must then systematically and thoughtfully provide a clear and easy to follow summary as well as critical analysis of the work done so far in terms of technical and scientific merits and missed opportunities.  Then you are to provide a clear step-by-step set of suggestions that will lead to strong innovative or interesting research directions for this project that is novel, insightful, and impactful for the JEPA, world models, or agentic research communities.

## Output Instructions

Your document output will be a well-structured, critically-written, well-craft documents in the folder: "docs/neurips-laterality".  

Your first document is a critique, and detailed step-by-step tutorial on how to thoughtfully extend the existing writings, notebooks and results.  It must be written in a clear, easy-to-follow, detailed step-by-step tutorial format with lots of explanations as well as illustrative vector graphics to clarify key concepts, ideas, and workflows.  You must use your best UI/UX design skills in designing and crafting these illustrative vector graphics and store them in the subfolders "figures".

To generate additional notebooks to point to the most promising research directions, you are to add to the jupyter notebooks in the toplevel observing the incremental naming convention of these notebooks.  Your notebooks must have clear, specific, easy-to-understand, grounded, and plenty of explanations and step-by-step tutorials to led the researcher forward with the most likely impactful research directions.

Then, you must implement your suggestions systematically and thoughtfully.

Finally, in the output folder ("docs/neurips-laterality"), ultrathink to write a paper and an extended abstract for this Physical World AI: Geometry, Characteristics, and Multimodal Sensing workshop with well argued contributions to the field using natural and fluent writings and supporting evidence.  Include appropriate results, figures, tables, numerical evidence, clear and convincing arguments and logical flow.

You must verify and check your claims, facts, and inferences to ensure that your response is accurate, correct, useful, and grounded. 

Use Codex adversarial review to review your thinking, inference, results, and writing carefully.  Fix all issues systematically and thoroughly.

Fan out subagents and use dynamic workflows.

---

Ultrathink how do we re-setup the dataset split, training, testing, and evaluation in order to remove the limitations noted?

"""
Results are transductive (internal validity only; no generalization claim). The dataset's official distribution provides annotations and public URLs, not raw video; we use derived pose sequences, infer no identity, and redistribute no raw or identity-bearing frames. No institutional ethics determination or completed data-use review is yet on record; both must be resolved before submission. Condition folder labels are dataset annotations, not diagnoses; the independent unit is the source video, not the individual; no clinical claim is made.
"""