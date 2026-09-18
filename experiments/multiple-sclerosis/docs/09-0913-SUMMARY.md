# Executive summary: learning from gait geometry and symmetry

*13 September 2026 · Retained results from notebooks 00–06; no new training experiments were run for this update.*

## Why geometry and symmetry matter

Walking depends on relationships between body parts, including how the knee bends as the hip moves and how the two sides alternate. Joint angles describe body geometry without changing when an image is shifted or uniformly enlarged. Comparing symmetry requires matching corresponding stages of a stride, so ordinary left–right alternation is not mistaken for an impairment.

Clinical measurements motivate these questions. A small early-Parkinson’s study found greater [arm-swing asymmetry](https://pubmed.ncbi.nlm.nih.gov/19945285/), while a multiple sclerosis study found altered [coordination between hip, knee, and ankle motion](https://pubmed.ncbi.nlm.nih.gov/35305428/). Our video experiments have yet to establish either association in this collection.

![Synthetic left and right movement curves before and after matching stride phase](../images/progress-0913/01_symmetry_cycle.svg)

*These synthetic curves illustrate a proposed analysis. Matching stride phase reveals the amplitude difference already present in both panels.*

The project tests **S-JEPA**, a skeleton-based Joint Embedding Predictive Architecture that predicts features of hidden body points from visible ones. These features are lists of numbers describing a sequence. Its connection to physical AI lies in testing which geometric relationships survive learning and improve prediction.

Our working **null hypothesis** is that these learned features offer no improvement over a simpler joint-angle classifier on unseen sources. This organizes the investigation retrospectively; the retained notebooks do not document a test of this null defined before examining results. The complete reference favors the angle-based classifier, while the corrected S-JEPA trainer still needs a full evaluation.

## Establish measurements and protect the test data — notebooks 00–02

Notebooks 00–01 account for 49 clips from 37 source videos, of which **47 clips from 35 sources** yield usable poses. Each retained frame contains 33 body landmarks with three values: horizontal position, vertical position, and estimated visibility. The collection’s normal, multiple sclerosis, and Parkinson’s disease labels do not establish verified diagnoses or unique participants.

Centering each pose on the pelvis and dividing both spatial coordinates by one torso length preserves its two-dimensional angles and distance ratios, up to numerical precision. Visibility stays unchanged. This removes body travel from the normalized input, while changing scale between frames can alter movement amplitudes. Filling missing positions and repeating frames to extend short clips also changes the observations.

Notebook 02 groups each 32-frame window into **264 tokens**: eight groups of four frames for each of 33 landmarks. This preserves the supplied values and temporal order within each joint, with tags identifying joint and time. Training copies use consistent rotations, translations, scaling, and optional mirroring across a window, preserving each pose’s shape; mirroring also exchanges left/right landmark identities. The learned features have no guaranteed preservation of distances or symmetry.

Source assignments are fixed before training windows are created. All clips from a source remain together in each train/test split, including during the initial learning stage that uses no health labels. Feature selection, scaling, and classifier fitting use training data alone, and both methods share the same saved splits. These safeguards prevent direct source leakage; creating 481 overlapping windows does not increase the number of independent observations.

## Test learning, then test prediction — notebooks 03–06

Changing which body points are hidden gives every joint opportunities to contribute. Notebook 03 runs **800 training steps**. At each step, it predicts hidden features for a batch of 32 windows and uses the prediction error to adjust the model’s weights—the numbers governing its predictions. Training windows are sampled repeatedly, with test sources excluded. Error falls and the learned features vary across examples.

Notebook 04 reloads the trained model and restarts the other training state for 400 further steps, with no score improvement on the same ten test clips. Notebook 05’s saved visualization run failed before producing feature plots.

Notebook 06 compares S-JEPA with a **Random Forest**, a classifier combining many branching decision rules. **Macro-F1** combines how often predictions for each label are correct with how many examples of that label are found, then averages equally across labels. Higher values are better.

![Three separate evaluations: further training scores 0.600 before and after; the 500-update single-split demonstration scores 0.644 for S-JEPA and 0.915 for Random Forest; the older five-split reference scores 0.438 and 0.667](../images/progress-0913/16_protocol_separation.svg)

*An “update” is one training step; a “fold” is one train/test split; a “probe” is a classifier fitted to fixed learned features. The demonstrations and older complete evaluation use different training procedures and cannot be combined.*

The complete five-split reference gives **0.438 for S-JEPA versus 0.667 for the angle-based classifier**, a difference of **−0.228**. It predates a correction to the prediction-error calculation. No retained uncertainty range or repeated-run analysis establishes the reliability of these differences, and the repeatedly inspected source videos are not verified independent participants.

## Use the controls to choose the next question

The simpler controls sharpen the interpretation. Among two classifiers tested for each input, the higher scores were **0.703** using average pose and pose variation, and **0.636** using landmark visibility. These inputs discard frame order, yet retain useful predictive information. Posture, movement amplitude, camera setup, and detection behavior remain possible contributors; their separate effects have not been measured.

The next experiments should add angle ranges, left–right differences, and stride-aligned coordination to angle means one component at a time, keeping test sources fixed and all choices within training data. Separate joint comparisons would test which hip, knee, or ankle measurements add predictive value. Timing, missing-data handling, and the duplicated ankle-range feature need correction before versioned comparisons of the current trainer, followed by uncertainty estimates that account for related clips and evaluation on new participants.

The [detailed progress report](08-0913-PROGRESS.MD) and [illustrated slide tutorial](../slides/slides.md) provide the supporting methods, results, and proposed experiments.
