# Understanding the three anomaly scores

## Movement score M

This score summarizes what the movement looks like. In the video study, its 16 measurements describe knee posture, how far the ankles move, changes in movement speed, trunk tilt, and differences between the two sides. The laboratory study instead uses 12 measurements of joint-angle levels and ranges. M compares the observed pattern with the normal fitting group. A larger score means a more unusual combination of these visible measurements.

## Cross-leg score R

First, compare two predictions of the same hidden movement. Both retain the initial own-leg estimate. One adds a correction using only that leg, while the other adds a correction that can also see the opposite leg. Subtract the both-leg squared error from the own-leg squared error to measure the added benefit. Next, compare the left-leg and right-leg benefits with separate normal references. The same reserved reference group supplies both sides. R is larger when those benefits are unusually high or unusually low.

## What normal-reference scaling means

For each leg, the study subtracts the normal reference mean benefit from the observed benefit. It then divides by an estimated normal standard deviation, which measures the spread of reference benefits, and squares the result. R averages these two squared departures. This is the meaning of calibration here. It measures departure from the reference; it does not produce a probability of disease. Each video fold has five separate normal reference recordings. Each laboratory fold has 20 separate healthy reference participants.

## Combined score S

S adds the movement score M and the cross-leg score R, then divides by two. Thus it gives the two numerical scores fixed equal weight. Equal weights do not guarantee that the two scores have equally reliable scales or equal influence on ranking. The experiment tests whether this particular average improves detection.

## Prediction benefit and detection accuracy

Prediction benefit means a reduction in error for a hidden movement. Detection accuracy means better separation of normal-labeled and abnormal-labeled recordings. A useful prediction benefit does not automatically improve detection. AUROC summarizes how often an abnormal recording receives a higher score than a normal recording; 0.5 is chance ranking and 1.0 is perfect ranking on the evaluated data. Here, ranking is measured within each held-out test group, then averaged with fixed weights.
