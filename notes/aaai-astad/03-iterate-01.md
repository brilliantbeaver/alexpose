# Role

You are a world models expert specializing in Joint Embedding Predictive Architecture, pose estimation and vision transformers

# Tasks

You are to carefully review the proposed research direction in "experiments/aaai/docs/draft00", ultrathink to significantly revise, enhance, and add to the notebooks, experiments and documentations in "experiments/aaai" in order to maximize the probability of research methodology and likely results, based in neurological reasoning, that would make the writing of a strong manuscript for the 4th Workshop on Automated Spatial and Temporal Anomaly Detection (ASTAD) which has a due date in a few days:  https://ismart.ece.mcgill.ca/astad4/.

**GAVD Datasets for processing**: This project and notebooks should try to use the GAVD dataset as much as possible to satisfy reviewers' demand for reasonably sufficient dataset sizes.  Ultrathink if the GAVD dataset satisfy the generalizability requirements -- and document clearly in the README.md file in the output folder.  The GAVD dataset has been classified in "data/gavd" and the normal gaits are in the "normal" subfolder, while the abnormal gaits to consider include:

* parkinsons
* stroke
* cerebral palsy
* myopathic
* stroke

All video downloads must be made to "data/videos-gavd" with thoughtful consideration on folder organizations.

Fully understand the data ingestion pipeline that uses the `alexpose` repo (/Users/pmui/dev/alexpose/ambient) with working notebooks on how to process gait videos from the GAVD dataset here.

Fully revise the slides in "experiments/aaai/slides" that contains illustrative slides documenting step-by-step with plenty of vector graphics, tables, and flowcharts explaining the setup and experiments.

Your language should be natural, full-sentences, easily accessible, direct and easy to understand, without using em-dashes, while connecting across ideas and paragraphs well.

Use critical adversarial-review to check and verify all of your work; systematically and thoughtfully fix all issues.

Use fan out subagents with dynamic workflows to parallize your tasks.

---

After you are done with your updates: systematically execute the experiments and notebooks with real data and real model calls.  Check and verify the results and outputs are ready and then update the "slides" and documents across "experiments/aaai".  

Based on the completed results on the real GAVD datasets and downloaded videos, create a strong manuscript at "experiments/aaai/docs/draft01" for the 4th Workshop on Automated Spatial and Temporal Anomaly Detection (ASTAD) which has a due date in a few days:  https://ismart.ece.mcgill.ca/astad4/.  In that folder, clearly document how ready the draft is for submission, including systemic critiques and concrete suggestions to improve the draft.  Based on these suggestions and critiques, ultrathink to iterate an improved draft thoughtfully, using grounded results from the experiments.  Use deep adversarial review to check and verify your outputs.  Fix all issues carefully and coherently.

Your instruction should include systematic update to all relevant documentations and slides.

---

