**Role**: You are a world-class AI researcher, machine learning scientist, and AI engineer specializing in world models, computer vision, and neural networks.

**Task**: You are to deeply and thoughtfully analyze the most important source code, notebooks, outputs, and writeups in this repo. The research direction and results are summarized in the 2 attached documents documenting my progress on the pose estimation and S-JEPA pipelines.

After mastering the technical concepts and workflow in this repo, carefully and systematically read and understand the following papers and tools:

* GoalForce: https://arxiv.org/pdf/2601.05848
* Masked Visual Actions: https://arxiv.org/pdf/2607.19343
* ControlNet: https://arxiv.org/pdf/2302.05543
* S-JEPA: https://sjepa.github.io/
* MuJoCo: https://mujoco.org/
* SAM 2: https://github.com/facebookresearch/segment-anything
* GEN-1.5: https://generalistai.com/blog/gen-1.5
* Sleep foundation model: https://www.nature.com/articles/s41591-025-04133-4
* GaitDynamics: https://www.nature.com/articles/s41551-025-01565-8
* AddBiomechanics: https://addbiomechanics.org/download_data.html
* Wan 2.2: https://github.com/Wan-Video/Wan2.2

Similar to the idea in the Masked Visual Actions paper of predicting the robot motion that could produce a desired object trajectory, ultrathink on creatively analyzing the above work and thoughtfully produce a potentially groundbreaking research direction in a similar scale to those projects. You should build on the work that I have already done in this repo, either directly or indirectly.

After relentlessly brainstorming, checking related recent and highly cited papers on arxiv and other authoritative sources, and playing around with various highly promising ideas, think extremely critically about how to align your thoughts with the following topics:

* How do you get a vision model to get good at understanding physics?
* How can we estimate whether a movement is physically grounded?
* Using S-JEPA for discriminative and generative purposes (generative could be SMPL keypoints)
* Using robot simulation pipelines or some estimation from a tool like OpenSim to correct a world model's loss during training
* Allowing depth estimation that is accurate using appropriate tools on top of the embedding that is already there in terms of movement understanding
* Building an agent with Qwen that learns how to pull all of the downstream tools and learn using RL (Tool calls: depth estimation, segment anything, monocular estimation using WHAM)
* Generating movement with joint keypoints and the keypoints of objects

Once you thoughtfully tackle these concepts, ultrathink a list of the top 7 innovative, concrete, and compelling ideas to extend the research for this sjepa (Skeleton JEPA) project and write a highly novel, unique, and significant paper on world models that is extremely relevant to the AI research community (ICLR/ICML/NeurIPS) as well as very useful for biomechanics researchers and scientists.

Your description & writeup for each proposed idea should be succinct, well-structured, compelling, and specific. It must describe in well-supported detail what the research question is, related works, and potential methods and experiments. The research question must be specific, measurable, achievable, relevant, and time-bound. Ultrathink on how to simplify the language and explanations to make each proposal much, much easier to understand. Start from first principles and fully explain all concepts and how they inter-relate with each other.

Illustrate frequently and abundantly with vector graphics and flowcharts to illuminate ideas and concepts, saving the images in the "images" subfolders. Use  adversarial review of your generated vector graphics and workflow to ensure that they are not cluttered, and that no excessive text or line overlaps. Each generated image should be easy to understand and with minimal clutter.

After your careful revision, thoughtfully consider how you could make the revision even better and clearer with concrete and specific suggestions. Based on these suggestions, thoroughly and systematically revise the draft to a final version that greatly improves the readability and conceptual clarity of the different proposals. Provide clear and referenceable citations to key papers from arxiv and other authoritative sources. Do not use em-dashes in your writing.

Fan out subagents.  Use dynamic workflow orchestration.
