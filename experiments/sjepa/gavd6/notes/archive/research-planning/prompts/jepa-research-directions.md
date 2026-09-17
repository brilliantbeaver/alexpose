**Role**: You are a senior research scientist in world models, JEPA, computer vision, and gait analysis, acting as the ideation lead for this repo.

**Task**: Your objective is seven decision-ready research proposals related to Joint-Embedding Predictive Architecture (JEPA). You are brainstorming and optimizing for one genuinely novel and significant contribution that can be demonstrated in under two weeks, not for coverage and not for a leaderboard win. 

Ultrathink on how to creatively extend the current set of experiments toward an insightful new research direction. You must be grounded in hard evidence, logical reasoning, compelling inferences, and insightful analysis of real data. 

Your main priority is to generate a high-probability route to an ICLR main track paper. Your proposed experiments must be able to produce significant, real results from real data (GAVD and AMASS). You cannot brainstorm ideas whose main contribution will not add significant value or knowledge to the AI/ML community. Think deeply about how to thoughtfully and logically come up with scientifically sound and innovative methods and experiments to support your claims. Only use existing models and techniques that are robust and verified.

Compute: 8x H100 GPUs.

Time: Total window is 1 week. A first result must be reachable within a few days of training and experimenting.

Assume implementation and experiment scaffolding are done rapidly by advanced coding agents such as Claude Code and Codex. Do not discount an idea for being fiddly to implement. Do discount it for needing GPU time, for needing data I do not have, or for depending on a checkpoint that is not publicly downloadable.

The papers below define the quality bar and serve as your inspiration for generating ideas. Your proposals should be comparable in ambition and in the size of the conceptual leap and impact:

- GoalForce: https://arxiv.org/pdf/2601.05848
- Masked Visual Actions: https://arxiv.org/pdf/2607.19343
- ControlNet: https://arxiv.org/abs/2302.05543
- S-JEPA: https://sjepa.github.io/
- Sleep foundation model: https://www.nature.com/articles/s41591-025-04133-4
- GaitDynamics: https://www.nature.com/articles/s41551-025-01565-8
- GAVD: https://arxiv.org/pdf/2407.04190
- GaitForeMer: https://arxiv.org/pdf/2207.00106

Then search for and read recent work from arxiv or other authoritative sources on world models, JEPA, and computer vision applied to gait analysis. After thoughtfully reviewing everything, come up with a list of the top 7 innovative, concrete and compelling ideas to build on and extend the research for this project.

Your description & writeup for each proposed idea should be clear, compelling, and specific. It must clearly describe the research question, related works, and potential methods and experiments. Make sure that your proposed ideas are truly novel and distinct from work that others have done. Your proposed ideas cannot simply be incremental improvements. They must make a real contribution, similar in scale to GoalForce and Masked Visual Actions. The research question must be specific, measurable, achievable, relevant, and time-bound. Think deeply on how to simplify the language and explanations to make each proposal much, much easier to understand. Start from first principles and fully explain all concepts and how they inter-relate with each other. You should be writing at an advanced high school level.

Illustrate frequently and abudantly with vector graphics and flowcharts to illuminate ideas and concepts, saving the images in the "images" subfolders. Use adversarial review of your generated vector graphics and workflow to ensure that they are not cluttered, and that no excessive text or line overlaps. Each generated image should be easy to understand and with minimal clutter.

After your careful revision, thoughtfully consider how you could make the revision for each proposal even better and clearer with concrete and specific suggestions. Based on these suggestions, thoroughly and systematically revise the draft to a final version that greatly improves the readability and conceptual clarity of the different proposals. Provide clear and referenceable citations to key papers from arxiv and other authoritative sources. Do not use em-dashes in your writing. Finally, rank each proposed idea based on novelty, projected significance, feasibility, and "wow factor".

Fan out subagents.  Use dynamic workflow orchestration.

**Output**: You should store all of your proposals in `notes/research-agenda`.
