# GaitDynamics: a generative foundation model for analyzing human walking and running

## Problem

Gait analysis models usually accept one fixed set of kinematic inputs and predict one fixed output. GaitDynamics asks whether one generative model can fill missing kinematics, estimate ground reaction forces, and predict the consequences of gait modifications across datasets and populations.

## Method in five sentences

1. The model represents each 1.5-second trial as 150 time steps by 75 OpenSim kinematic and force parameters.
2. A four-block diffusion transformer learns to denoise complete windows and can inpaint any hidden combination of kinematics or forces.
3. A separate six-block transformer maps complete kinematics to ground reaction forces after the diffusion model fills missing kinematics.
4. Training uses study-balanced and trial-balanced sampling from the pruned AddBiomechanics corpus, with 100,000 diffusion steps and 300,000 refinement steps.
5. At inference, known parameters are clamped during fifty denoising iterations, which permits force estimation, gait modification, and speed-conditioned generation with one checkpoint.

## Headline numbers

**[preprint full text]** Pruning yielded 34.8 hours from 270 participants, with 10,352 training trials from 178 people and 1,929 test trials from 92 people. With full-body kinematics, mean absolute errors were 3.2, 1.2, and 0.7 percent of body weight for vertical, anterior-posterior, and medial-lateral force profiles, and 3.6 percent for peak vertical force. A 1.5-second sample took under one second on an RTX 3050 laptop GPU. Training took 30 hours on one RTX A6000. The model's running-speed interventions produced a 39 percent rise in peak anterior-posterior force and an 8 percent rise in peak vertical force from 3 to 5 m/s, compared with measured rises of 48 and 10 percent.

## What it makes possible here

The released diffusion checkpoint is a direct normal-gait prior and a stronger baseline than the repository's small S-JEPA for metric gait dynamics. GAVD video would need lifting and retargeting into its OpenSim schema. AMASS could test that retargeting and provide synthetic 3D gait interventions. A decisive study could compare diffusion surprise, minimum normalizing edit, and predicted force consistency as distinct measurements of pathology. GaitDynamics itself does not classify pathology from video.

## Limitations

**[preprint full text]** Inputs use laboratory-derived metric OpenSim kinematics, not noisy monocular RGB. The force head was trained on synchronized force-plate data, so its predictions are a learned prior rather than measurements on GAVD. Training data include no pathological gait; osteoarthritis accuracy held, but one post-stroke case showed a drop. The model omits muscles. It prunes turns, short trials, high trunk rotation, and implausible center-of-pressure records. Its two intervention validations use studies represented in the training distribution of gait patterns.

## Access status

The final Nature page was blocked by sign-in on 2026-09-03. Its abstract was fetched and is marked **[abstract only]** when used. The full March 2025 Research Square preprint was read, so detailed claims above are marked **[preprint full text]**. Public GitHub code includes directly downloadable 137 MB diffusion and 8.01 MB force-refinement checkpoints. The repository also provides a Colab and Hugging Face demo. The model expects the OpenSim Rajagopal model without arms, so joint-schema mismatch can cause failure.

## Sources

- Final article page and abstract: https://www.nature.com/articles/s41551-025-01565-8
- Full preprint: https://pmc.ncbi.nlm.nih.gov/articles/PMC11957236/
- Preprint PDF: https://assets-eu.researchsquare.com/files/rs-6206222/v1_covered_bd1a4fd8-bd9f-402b-a1ea-d951837522b9.pdf?c=1767687457
- Official code and checkpoints: https://github.com/stanfordnmbl/GaitDynamics
