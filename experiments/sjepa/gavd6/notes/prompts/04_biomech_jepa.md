**Role**: You are a senior research scientist in world models, JEPA, computer vision, and gait analysis, acting as the ideation lead for this repo.

**Task**: Your objective is seven decision-ready research proposals related to Skeleton JEPA (S-JEPA) for gait/balance analysis or next action prediction from kinematics. You are brainstorming and optimizing for one genuinely novel and significant contribution that can be demonstrated in under two weeks, not for coverage and not for a leaderboard win.

Carefully and thoughtfully read through the attached documents to understand the research directions tried so far. Next, you are to synthesize the methods, experiments, and results from notebooks 00 to 06 in `gavd6/notebooks/foundations`, as well as the latest experiments in `gavd6/outputs`. Also look through the rest of sjepa to explore more ideas and directions. Ultrathink on how to creatively extend the experiments toward an insightful new research direction. You must be grounded in hard evidence, logical reasoning, compelling inferences, and insightful analysis of real data.

The primary gait dataset for your proposal, if needing to classify normal and abnormal gait, should be GAVD (https://github.com/Rahmyyy/GAVD, paper https://arxiv.org/abs/2407.04190). It contains 1,874 clinically annotated in-the-wild RGB sequences, 400+ subjects, spanning normal, abnormal, and pathological gait, in settings from clinics to uncontrolled urban video. This is where classification happens. The manifests are stored in `manifests/gavd`. The full videos are stored on my personal scratch space on HAIC.

If GAVD is not sufficient for your idea proposals, search extensively for publicly available, compatible, and well-known datasets such as AMASS (which I have already stored on HAIC). However, do not be restricted by just the datasets I have available. If you think of an idea with high potential that works best with another dataset that fits my above criteria, use that instead. If you decide to use AMASS, also think about expanding from just using the lower body keypoints (Core11) to the whole body.

Training S-JEPA models from scratch can be a starting point for your proposed research directions, but they cannot be the main contribution. There are already large pretrained world models such as V-JEPA and GaitDynamics that used much more compute than I have access to. One approach you can take is to imitate the idea used by GoalForce and Masked Visual Actions: take a large existing model, in their case Wan 2.2 (https://github.com/Wan-Video/Wan2.2), and adapt it. In the case of Masked Visual Actions, the team fine tuned Wan 2.2 on only 15 hours of robot data and were able to get outstanding results.

Some possible techniques include using LoRA and adapters, ControlNet-style conditioning branches, frozen-encoder probing, distillation into small heads, test-time optimization, and using a pretrained model as a scorer or simulator rather than training it at all. The ambition should be large; the training should not be.

Compute: 8x H100. Calibration anchor: a 100-epoch JEPA training run on AMASS took roughly 3 hours on a single H100 GPU compute node.

Time: Total window is 1 to 2 weeks. A first result must be reachable within a few days of training.

Assume implementation and experiment scaffolding are done rapidly by advanced coding agents such as Claude Code and Codex. Do not discount an idea for being fiddly to implement. Do discount it for needing GPU time, for needing data I do not have, or for depending on a checkpoint that is not publicly downloadable.

Known dead ends: Do not propose these:
- Binary normal-versus-abnormal detection on GAVD as the headline result. Generic action recognition already reaches roughly 94% (TSN) and 92% (SlowFast) on that task. It is saturated and beating it proves nothing.
- Anything requiring force plates, synchronized instrumented gait measurement, or clinical data collection.
- Anything whose headline metric could be produced by a trivial signal. My prior audits on related data found that clip duration, centroid drift, and foreground area recover supposedly semantic gait factors at high accuracy. Assume the reviewer knows this.

The papers below define the quality bar and serve as your inspiration for generating ideas. Your proposals should be comparable in ambition and in the size of the conceptual leap and impact:

- GoalForce: https://arxiv.org/pdf/2601.05848
- Masked Visual Actions: https://arxiv.org/pdf/2607.19343
- ControlNet: https://arxiv.org/abs/2302.05543
- S-JEPA: https://sjepa.github.io/
- Sleep foundation model: https://www.nature.com/articles/s41591-025-04133-4
- GaitDynamics: https://www.nature.com/articles/s41551-025-01565-8
- GAVD: https://arxiv.org/pdf/2407.04190
- GaitForeMer: https://arxiv.org/pdf/2207.00106

Then search for and read recent work from arxiv or other authoritative sources on world models, JEPA, and computer vision applied to gait analysis. After thoughtfully reviewing everything, come up with a list of the top 7 innovative, concrete and compelling ideas to build on and extend the research for this sjepa (Skeleton JEPA) project.

Your description & writeup for each proposed idea should be clear, compelling, and specific. It must clearly describe the research question, related works, and potential methods and experiments. Make sure that your proposed ideas are truly novel and distinct from work that others have done. Your proposed ideas cannot simply be incremental improvements. They must make a real contribution, similar in scale to GoalForce and Masked Visual Actions. The research question must be specific, measurable, achievable, relevant, and time-bound. Think deeply on how to simplify the language and explanations to make each proposal much, much easier to understand. Start from first principles and fully explain all concepts and how they inter-relate with each other. You should be writing at an advanced high school level.

Illustrate frequently and abudantly with vector graphics and flowcharts to illuminate ideas and concepts, saving the images in the "images" subfolders. Use adversarial review of your generated vector graphics and workflow to ensure that they are not cluttered, and that no excessive text or line overlaps. Each generated image should be easy to understand and with minimal clutter.

After your careful revision, thoughtfully consider how you could make the revision for each proposal even better and clearer with concrete and specific suggestions. Based on these suggestions, thoroughly and systematically revise the draft to a final version that greatly improves the readability and conceptual clarity of the different proposals. Provide clear and referenceable citations to key papers from arxiv and other authoritative sources. Do not use em-dashes in your writing. Finally, rank each proposed idea based on novelty, projected significance, feasibility, and "wow factor".

Fan out subagents.  Use dynamic workflow orchestration.

**Output**: You should store all of your proposals in `notes/world-model-extensions/proposals-03`.
