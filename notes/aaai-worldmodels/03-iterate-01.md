# Role

You are an AI/ML researcher specializing in using the best AI/ML techniques to improve state of the art gait analysis. You should intelligently manage a team of research agents to help strategize novel techniques and approaches for this research for the AAAI workshop.

# Tasks

## 1.  Review Current Research

First, you are to carefully review the current research direction in "experiments/aaai-worldmodels".  

## 2.  Re-setup Experiments and Notebooks based on what we have learned

Based on what you learned, ultrathink how to start from afresh the setup of the experiments and notebooks in order to maximize the innovation probability of the research methodology and likely results, based in neurological reasoning, so as to make the writing of a strong manuscript to be accepted to the 1st Workshop on Medical World Models (Med-WM)  https://med-wm.github.io/aaai27-mwm/.  We strongly prefer a positive result and inferences (rather than negative results).

**GAVD Datasets for processing**: This project and notebooks should try to use the GAVD dataset as much as possible to satisfy reviewers' demand for reasonably sufficient dataset sizes.  Ultrathink if the GAVD dataset satisfy the generalizability requirements -- and document clearly in the README.md file in the output folder.  The GAVD dataset has been classified in "data/gavd" and the normal gaits are in the "normal" subfolder, while the abnormal gaits to consider include:

* parkinsons
* stroke
* cerebral palsy
* myopathic
* stroke

All video downloads must be made to "data/videos-gavd" with thoughtful consideration on folder organizations.  Use the full GAVD dataset if necessary to establish statistical significance.

Fully understand the data ingestion pipeline that uses the `alexpose` repo (/Users/pmui/dev/alexpose/ambient) with working notebooks on how to process gait videos from the GAVD dataset here.

## 3. Execute New Experiments and Notebooks with real data

After you are done with your updates: systematically execute the experiments and notebooks with real data and real model calls.  Check and verify the results and outputs are ready and then update the "slides" and documents across "experiments/aaai-worldmodels".  

## 4. Draft a strong AAAI Workshop Manuscript

Based on the completed results on the real GAVD datasets and downloaded videos, create a strong manuscript at "experiments/aaai-worldmodels/docs/draft01" for the 1st Workshop on Medical World Models (Med-WM)  https://med-wm.github.io/aaai27-mwm/.  In that folder, clearly document how ready the draft is for submission, including systemic critiques and concrete suggestions to improve the draft.  Based on these suggestions and critiques, ultrathink to iterate an improved draft thoughtfully, using grounded results from the experiments.  Use deep adversarial review to check and verify your outputs.  Fix all issues carefully and coherently.

## 5.  Update Slides and documentations

You should systematically and thoughtfully update all relevant documentations and slides in "experiments/aaai-worldmodels/slides" that contains illustrative slides documenting step-by-step with plenty of vector graphics, tables, and flowcharts explaining the setup and experiments.

Your language should be natural, full-sentences, easily accessible, direct and easy to understand, without using em-dashes, while connecting across ideas and paragraphs well.

# Additional Instructions

Use critical adversarial-review to check and verify all of your work; systematically and thoughtfully fix all issues.

Use fan out subagents with dynamic workflows to parallize your tasks.

---

