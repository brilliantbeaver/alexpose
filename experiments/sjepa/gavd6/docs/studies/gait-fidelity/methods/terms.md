# Terms and examples for the movement-response study

[Methods index](README.md) · [Response protocol](jepa-response.md) · [Evaluation](evaluation.md)

The study follows a simple sequence. A video supplies estimated joint positions. A model attempts to correct their errors. We then measure whether a known change between two movement states survives those corrections. The reference tells us what the change should have been; it is available for training and evaluation, but it is not supplied to the deployed model.

## Poses and changes in movement

A **pose** is a set of joint positions at one time. A **trajectory** follows those positions over time. The **body12** representation has twelve joints: left and right shoulders, elbows, wrists, hips, knees and ankles. A **coordinate** is a numerical position, such as a knee being 120 pixels from the left edge and 200 pixels from the top of an image.

An **observation** is the estimated trajectory supplied to the model. It can contain errors or missing joints. A **reference** is the independently specified trajectory used for comparison. In this synthetic study, the reference comes from the source motion, body model and camera projection. It has a defined geometric convention, rather than being a perfect clinical measurement of a person.

**Restoration** means correcting observed trajectories. A **target** is a value that training asks the model to predict; depending on the training objective, it may be a reference coordinate or a feature extracted from that reference.

A **movement pair** contains a baseline sequence and another movement state derived from the same source. Each sequence is an **endpoint** of the comparison. These endpoints are not adjacent video frames. A **response** is the difference in a specified measurement across those states. For example, if the reference right-minus-left knee excursion goes from +5° to 0°, the reference response is −5°. A restored response of −2° has an absolute response error of 3°.

**Knee excursion** is the 95th percentile minus the 5th percentile of the projected knee angle over the retained interval. A percentile identifies a value below which a stated fraction of samples falls. These percentiles summarize most of the angular range without relying on a single extreme frame. **Projected** means measured in the two-dimensional camera image; projected degrees are not automatically anatomical three-dimensional range of motion.

A **nuisance condition** changes the observations, such as hiding part of the image or exchanging estimated left/right names. It is distinct from the movement change being measured. **Occlusion** means that an object or body part blocks visibility. A camera also changes projected geometry, so each camera needs its own reference.

## What the model learns

A **feature vector**, also called a **latent representation**, is a learned list of numbers used inside the model. Its components do not come with physical units such as pixels or degrees. A feature difference therefore needs evidence before it can be interpreted as useful movement information.

The **encoder** converts observed trajectories into features. The **predictor** transforms those features toward reference features. The **teacher** extracts the reference features used as targets. In this implementation, its learned parameters follow those of the student encoder gradually through an **exponential moving average**, a weighted update that retains most of the previous teacher and incorporates a small part of the current student.

**JEPA** means joint-embedding predictive architecture. Here it learns to predict reference features from observed trajectories. The initial learning stage is **pretraining**. A later **readout** learns coordinate corrections from the encoder's features.

A **parameter**, or weight, is a learned numerical setting in a model. A **frozen encoder** keeps those settings fixed while the readout learns. Its output features still change when its input changes; freezing does not force every movement to have the same representation.

A **token** groups four consecutive frames of one joint. An artificial **mask** marks tokens hidden during pretraining. A **query** marks a position selected for prediction; queries include artificial hiding and naturally missing observations. **Context** is the remaining observation information available to the encoder. A **graph-time mask** chooses anatomically connected joint groups and time intervals to hide.

**Normalization** expresses coordinates relative to an origin and scale. Here each sequence uses one observation-derived origin and one scale for both axes and all frames. Hidden observations are excluded when pretraining computes those values. This avoids using hidden or reference information to define the student's transform.

## Losses, fitting and calibration

A **residual** is prediction minus target. If the prediction is 7 and the target is 5, the residual is +2. A **loss** is the number minimized during training, such as a squared residual. An **auxiliary loss** adds a particular training requirement to the existing base loss. A **coefficient** sets how much that term contributes.

The delta JEPA auxiliary compares the two endpoints' residuals. The endpoint JEPA auxiliary penalizes each residual separately. If both residuals are +2, their difference is zero although both endpoints are wrong. This is why a small difference loss must be checked against level and coordinate accuracy.

A **gradient** is a derivative that describes how the loss changes when model parameters change. An **optimizer** uses gradients to update those parameters. **Calibration** here means setting the auxiliary coefficients by a fixed rule using gradients from training batches before optimization begins. It is separate from assessing a clinical instrument's calibration. **Clipping** limits a gradient's total size to prevent an excessively large update.

A **batch** is a group of examples processed together in one update. A **seed** sets reproducible random choices such as initialization and example draws. A **checkpoint** saves the learned parameters and training state. A **receipt** records artifact identities and settings so the saved run can be checked for changes.

A target is **detached** when a loss cannot send gradients into the branch producing it. The teacher is detached here. Its parameters change through the moving-average rule, rather than through the auxiliary loss's backward gradient.

## Reading the evidence

**Support** is the set of positions with the required references and queries. **Reference eligibility** determines whether enough reference-supported measurements exist to score an example. **Coverage** reports how many examples were eligible and how many eligible predictions succeeded. A **conditional** score uses only a stated subset, such as successful outputs. The primary response error retains failed eligible predictions using a declared penalty, so it must be read alongside conditional plots.

A **source family** groups related versions of one source window. A **window** is a fixed interval from a recording. Different cameras, corruptions and windows from one person remain correlated observations. **Person-balanced** aggregation gives people equal weight after averaging their repeated observations; thousands of rendered records do not become thousands of independent people.

A **bootstrap** repeatedly resamples the observed units to estimate uncertainty. This study resamples people and fitted seeds as separate, **crossed factors**, because every fitted seed is evaluated on every person. Compared methods remain paired within each person and seed. An uncertainty interval crossing zero does not prove that methods are equivalent.

A **probe** is a small prediction model used to inspect information in a representation. The fixed **ridge probe** is linear regression with a penalty on large coefficients. **Person-separated folds** keep every observation from a person together while different groups take turns being held out. This diagnostic tests whether a particular linear readout can recover the response; a failed probe does not rule out nonlinear information.

**Development evidence** comes from people used in the study's development process. **Confirmation** would require a separately admitted, previously unopened population and a comparison fixed before its outcomes were read. The active follow-up uses development evidence only. A **CPU fixture** uses small generated data to check software behavior; the H100 GPU source experiment is still needed to obtain scientific results.
