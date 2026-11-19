# Role

You are a world class AI/ML researcher specializing in using the best AI/ML techniques to improve state of the art gait analysis. You should intelligently manage a team of research agents to help strategize novel techniques and approaches for this research for the AAAI workshop.

# Tasks

You are to first carefully review the documents in "experiments/aaai-worldmodels/docs/draft01" as well as results in the "experiments/aaai-worldmodels" notebooks, ultrathink with out-of-the-box creative thinking how best to significantly enhance, and add to the notebooks, experiments and documentations in "experiments/aaai-worldmodels" in order to maximize the acceptance probability of the manuscript based on the research methodology and likely results, neurological reasoning for the 1st Workshop on Medical World Models (Med-WM)  https://med-wm.github.io/aaai27-mwm/

You are then to write the instruction for the next iteration of research methodology, notebook design, and technical approaches as a systematically thoughtful set of instructions in "notes/aaai-worldmodels/05-iterate-02-INSTR.md".  

Your **primary directive** is to maximize the probability of novel research methodology, results, based in neurological reasoning, that would make the writing of a strong manuscript for the 1st Workshop on Medical World Models (Med-WM)  https://med-wm.github.io/aaai27-mwm/

**GAVD Datasets for processing**: This project and notebooks should try to use the GAVD dataset as much as possible to satisfy reviewers' demand for reasonably sufficient dataset sizes.  Ultrathink if the GAVD dataset satisfy the generalizability requirements -- and document clearly in the README.md file in the output folder.  The GAVD dataset has been classified in "data/gavd" and the normal gaits are in the "normal" subfolder, while the abnormal gaits to consider include:

* parkinsons
* stroke
* cerebral palsy
* myopathic
* stroke

All video downloads must be made to "data/videos-gavd" with thoughtful consideration on folder organizations.  Use the full GAVD dataset if necessary to establish statistical significance.

Fully understand the data ingestion pipeline that uses the `alexpose` repo (/Users/pmui/dev/alexpose/ambient) with working notebooks on how to process gait videos from the GAVD dataset here.

Fully revise the slides in "experiments/aaai-worldmodels/slides" that contains illustrative slides documenting step-by-step with plenty of vector graphics, tables, and flowcharts explaining the setup and experiments.

Your language should be natural, full-sentences, easily accessible, direct and easy to understand, without using em-dashes, while connecting across ideas and paragraphs well.

Use critical adversarial-review to check and verify all of your work; systematically and thoughtfully fix all issues.

Use fan out subagents with dynamic workflows to parallize your tasks.

# Additional Instructions

The output instruction should NOT need to mention "Decision date".  Instead, the output should focus on what will make the next best set of novel experiments.

Based on deep research on alternative masking techniques, ultrathink how to modify the instruction to experiment with the state of the art masking alternatives.

Your output instruction should follow the best practices for prompting any frontier language models -- perhaps starting with a "role" definition and clear goal and tasks definitions.

By reviewing the experimental setup and results so far, think creatively and out-of-the-box, ultrathink if there is a positive result interpretation and writeup.  If possible, draft using plain language and complete sentences to "draft02"

draft02 should not reference any of the "original analysis remains negative".  This manuscript should be a standalone writeup of a significant contribution to anomaly detection for gait analysis.

Carefully and thoughtfully rewrite your draft02 to make the reading more accessible and less dense with jargons.  The reader audience should be generally AI/ML researchers but not specialists about JEPA, self-supervised learning, or gait analysis.  Use simple complete sentences with clear connection words.  Proof read so that the main text, captions, and image illustrations are easy to understand.  Avoid syllogism and staccato cadence

---

Add to the notes/aaai-worldmodels/05-iterate-02-INSTR.md instructions for writing a positive results-based manuscript for AAAI [Med-WM workshop](https://med-wm.github.io/aaai27-mwm/) as the next draft03, using plainly written, grounded, easy to read, fully explained, well composed, logically connected languages and sentences.  Explain all key concepts and ideas clearly while able to emphasize our contributions.
