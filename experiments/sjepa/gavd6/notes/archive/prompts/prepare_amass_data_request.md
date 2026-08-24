**Role**: You are an expert world model (JEPA) researcher well versed in human gait analysis with world models

**Task** You are to carefully and systematically implement a clean and efficient script to convert the AMASS data (not stored in this environment, stored in my personal HAIC scratch space), which is formatted as `poses.npz` files, into the shared skeleton schema specified in `gavd6/notes/ideas-claude/05-09-gait-parity`.

You must implement the following tasks. Do not write unnecessary code that is not helpful to the goal of the script, which is to turn the AMASS data into usable joint coordinates for JEPA pretraining:

* load poses, trans, gender, frame rate, and the required body-model parameters;
* run the appropriate SMPL-family forward kinematics;
* select and order the core-11 joints;
* preserve anatomical left and right names;
* transform coordinates into the frozen body frame;
* resample to the frozen canonical frame rate;
* save coordinates and validity alongside provenance.

Use adversarial review to review and check all of your work, systematically and thoughtfully fix all issues.

Use fan out subagents with dynamic workflows to parallelize your tasks.
