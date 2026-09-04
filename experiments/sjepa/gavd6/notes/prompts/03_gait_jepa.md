**Role**: You are a senior research scientist in world models, JEPA, computer vision, and gait analysis, acting as the ideation lead for this repo.

**Task**: Your objective is seven decision-ready research proposals related to one specific direction: using Skeleton JEPA (S-JEPA) to classify and characterize human gait. You are brainstorming and optimizing for one genuinely novel and significant contribution that can be demonstrated in under two weeks, not for coverage and not for a leaderboard win.

Carefully and thoughtfully read through the attached documents to understand the research directions tried so far. Next, you are to synthesize the methods, experiments, and results from notebooks 00 to 06 in `notebooks/foundations`. Ultrathink on how to creatively extend the experiments toward an insightful new research direction. You must be grounded in hard evidence, logical reasoning, compelling inferences, and insightful analysis of real data.

The primary gait dataset for your proposals should be GAVD (https://github.com/Rahmyyy/GAVD, paper https://arxiv.org/abs/2407.04190). It contains 1,874 clinically annotated in-the-wild RGB sequences, 400+ subjects, spanning normal, abnormal, and pathological gait, in settings from clinics to uncontrolled urban video. This is where classification happens. The manifests are stored in `manifests/gavd`. The full videos are stored on my personal scratch space on HAIC.

Training S-JEPA models on GAVD can be a starting point for your proposed research directions, but they cannot be the main contribution. There are already large pretrained world models such as V-JEPA and GaitDynamics that used much more compute than I have access to. One approach you can take is to imitate the idea used by GoalForce and Masked Visual Actions: take a large existing model, in their case Wan 2.2 (https://github.com/Wan-Video/Wan2.2), and adapt it. In the case of Masked Visual Actions, the team fine tuned Wan 2.2 on only 15 hours of robot data and were able to get outstanding results.

Some possible techniques include using LoRA and adapters, ControlNet-style conditioning branches, frozen-encoder probing, distillation into small heads, test-time optimization, and using a pretrained model as a scorer or simulator rather than training it at all. The ambition should be large; the training should not be.

Compute: 8x H100. Calibration anchor: a 100-epoch JEPA training run on AMASS took roughly 3 hours on a single H100 GPU compute node.

Time: Total window is 1 to 2 weeks. A first result must be reachable within 36 hours of starting on a reduced setting if necessary.

Assume implementation and experiment scaffolding are done rapidly by advanced coding agents such as Claude Code and Codex. Do not discount an idea for being fiddly to implement. Do discount it for needing GPU time, for needing data I do not have, or for depending on a checkpoint that is not publicly downloadable.

Known dead ends: Do not propose these:
- Binary normal-versus-abnormal detection on GAVD as the headline result. Generic action recognition already reaches roughly 94% (TSN) and 92% (SlowFast) on that task. It is saturated and beating it proves nothing.
- Anything requiring force plates, synchronized instrumented gait measurement, or clinical data collection.
- Anything whose headline metric could be produced by a trivial signal. My prior audits on related data found that clip duration, centroid drift, and foreground area recover supposedly semantic gait factors at high accuracy. Assume the reviewer knows this.

Because binary detection is closed, the interesting territory is elsewhere: pathology type and laterality, severity as a continuous quantity, sample efficiency in the few-shot regime that clinical labels actually live in, generalization from clinic video to in-the-wild video and back, calibrated uncertainty under distribution shift, and interpretable or counterfactual output that a clinician could read. Aim there.

The papers below define the quality bar and serve as your inspiration for generating ideas. Your proposals should be comparable in ambition and in the size of the conceptual leap, not smaller:

- GoalForce: https://arxiv.org/pdf/2601.05848
- Masked Visual Actions: https://arxiv.org/pdf/2607.19343
- ControlNet: https://arxiv.org/abs/2302.05543
- S-JEPA: https://sjepa.github.io/
- Sleep foundation model: https://www.nature.com/articles/s41591-025-04133-4
- GaitDynamics: https://www.nature.com/articles/s41551-025-01565-8
- GAVD: https://arxiv.org/pdf/2407.04190
- GaitForeMer: https://arxiv.org/pdf/2207.00106

Then search for and read recent work from arxiv or other authoritative sources on world models and clinical gait analysis. After thoughtfully reviewing everything, come up with a list of the top 7 innovative, concrete and compelling ideas to build on and extend the research for this sjepa (Skeleton JEPA) project.

Your description & writeup for each proposed idea should be succinct, well-structured, compelling, and specific. It must describe in well-supported detail what the research question is, related works, and potential methods and experiments. The research question must be specific, measurable, achievable, relevant, and time-bound. Think deeply on how to simplify the language and explanations to make each proposal much, much easier to understand. Start from first principles and fully explain all concepts and how they inter-relate with each other. You should be writing at an advanced high school level.

Illustrate frequently and abudantly with vector graphics and flowcharts to illuminate ideas and concepts, saving the images in the "images" subfolders. Use adversarial review of your generated vector graphics and workflow to ensure that they are not cluttered, and that no excessive text or line overlaps. Each generated image should be easy to understand and with minimal clutter.

After your careful revision, thoughtfully consider how you could make the revision even better and clearer with concrete and specific suggestions. Based on these suggestions, thoroughly and systematically revise the draft to a final version that greatly improves the readability and conceptual clarity of the different proposals. Provide clear and referenceable citations to key papers from arxiv and other authoritative sources. Do not use em-dashes in your writing.

Fan out subagents.  Use dynamic workflow orchestration.

**Output**: You should store all of your proposals in `notes/world-model-extensions/proposals-02`.
