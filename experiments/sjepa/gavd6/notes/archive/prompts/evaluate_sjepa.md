**Role**: You are a world models expert specializing in Joint Embedding Predictive Architecture (JEPA), self-supervised learning, and pose estimation.

**Task**: You are to carefully and systematically add a frozen preprocessing adapter and downstream probe based on the checkpoints in `outputs/repaired-jepa-seed7-v2`, without retraining either JEPA encoder.

Create a Jupyter notebook as well as a Python script to run the downstream probe on the 96 GAVD videos loaded in previous experiments in the `notebooks/experiments` folder.

The current incompatibilities are:
* Checkpoints expect [64, 11, 3] Core11 body-frame input; GAVD stores variable-length [T, 33, 4] MediaPipe image-space poses.
* Core11 channels are forward/up/mediolateral, with reflection on channel 2. GAVD currently uses image x/y/z, with reflection on channel 0.
* Some stroke videos are 23.976 fps; the checkpoints were trained at canonical 30 fps.
* The existing classifier notebook expects a 33-joint, width-96 encoder and 384-dimensional pooling. These checkpoints use width 64 and paired orbit branches.
* JEPA has no Parkinson’s/stroke classification head. Masked-prediction loss alone is not downstream classification performance.

Ultrathink on how to build a valid adapter that would:
1. Map MediaPipe landmarks to pelvis, bilateral hips, knees, ankles, heels, and forefeet.
2. Construct the same pelvis-centred, leg-length-normalized forward/up/mediolateral frame used by AMASS.
3. Resample to 30 fps and create 64-frame windows with stride 32.
4. Freeze the EMA target_encoder.
5. Pool even/odd orbit features into a preregistered sequence vector, probably mean and standard deviation of each channel, yielding 256 features.
6. Fit identical nested linear/ridge probes for both checkpoints, with raw-coordinate and random-encoder baselines.

Make sure to only implement code that is directly relevant to evaluating the JEPA checkpoints on the GAVD dataset videos. Do not add unnecessary code for robust engineering, provenance, or reproducibility checks. This is a research-focused experiment.

Use independent adversarial review subagents to review and check all of your work. Systematically and thoughtfully fix all suggested issues.

Use fan out subagents with dynamic workflows to parallelize your tasks.
