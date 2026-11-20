# How to study walking with video

Imagine a camera records one walk from the side and another from the front. Could a computer recognize the same walking pattern in both? What if it learns that one camera angle tends to appear with one gait label? This tutorial follows the researchers as they design experiments that can tell those possibilities apart. You only need averages, fractions, and a little curiosity about how people move.

**The short answer.** A model that knew a walker's current posture predicted their next few leg positions more accurately than a rule that kept every joint still. Recognizing an unusual gait was harder. In a separate test, a classifier trained to read S-JEPA features earned the same balanced accuracy as one trained to read features from an encoder with random weights. The lessons explain how to check both claims yourself.

## Start with one recording

Gait is the pattern a person makes while walking. A **landmark** is a point the computer estimates on the body, such as a knee. A **pose** is the collection of those points in one frame. A landmark can disappear behind the other leg. The experiment keeps track of which coordinates were actually observed.

A **recording** is one source video. A **sequence** is a labeled stretch of walking in that video. Researchers sample short **windows** from a sequence so a model can work with a fixed number of observations. If you watch three windows from one sequence, you have three views of one walk. Counting them as three different people would give a misleading impression of how much independent evidence you have.

## Follow the investigation

| Lesson | What you will do | What to look for |
| --- | --- | --- |
| Learn from hidden joints (00–07) | Hide part of a pose and follow how the computer learns to describe it. | Can a classifier read a gait label when it sees a new recording? |
| Predict the next few steps (10–16) | Make a simple forecast, calculate its error, and compare it with copying the last pose. | Is the next position easier to predict because the model learned timing, or because the person moved very little? |
| Check whether errors flag unusual gait (20–26) | Compare scores within groups, then inspect results on a different dataset. | How often does a score rank an abnormal recording above a normal one? |

Read a lesson, try its small experiment, then open the numbered notebook if you would like to see the code. The examples you can change are teaching examples. The measured study values come from saved results and stay fixed.

## Open your first research notebook

1. Start with a numbered notebook link in a lesson. A notebook mixes explanations with short blocks of Python called cells. A cell may load observations, calculate a result, or draw a figure. Read the question above the cell before looking at the code.
2. Choose one output and identify its unit. For example, a row might describe a joint, a video window, a whole recording, or a participant. Ask which rows could have come from the same person or recording.
3. Find the comparison. If a forecast has an error of 0.08, you still need the error from a simpler rule on the same observations. Write down both values before deciding whether the forecast helped.
4. Write one sentence that names the method, the comparison, the number of independent examples, and what changed. Look for a result that would make you revise your sentence.

With a local copy of the repository, you can optionally start JupyterLab from the repository root. The project uses Python 3.12 and the uv package manager.

```bash
uv sync --project experiments/aaai-worldmodels --python 3.12
uv run --project experiments/aaai-worldmodels jupyter lab experiments/aaai-worldmodels
```

In JupyterLab, open a numbered notebook and press **Shift + Enter** to run a cell. Some cells need local videos, pose weights, and prepared model files. If a file is missing, you can still follow the lessons, inspect the code, and check the public result tables. The [experiment guide](../../../experiments/aaai-worldmodels/README.md) identifies the inputs needed for a full replay. The [research notes](../../../notes/aaai-worldmodels/) record the questions the team wanted to explore.

## Read the conference poster alongside the lessons

The left-hand S-JEPA illustration on the poster follows a separate study of 96 clips from 18 videos. The central test on 17 unseen recordings belongs to notebooks 00–07. The table of 91 recordings and the ankle forecast belong to notebooks 20–26. These comparisons use different training tasks. Keeping the cohort and the question together will help you read each number correctly.
