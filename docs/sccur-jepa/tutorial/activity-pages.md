# Activities to try on paper

## Follow a source video across a split

Two recordings, A and B, each supply two clips. Put A1 and B1 in training and A2 and B2 in testing. Circle every test clip whose original recording also appears in training. Now move both clips from A into training and both from B into testing. How many test clips come from familiar recordings in each arrangement?

**Check your reasoning.** Both test clips have familiar source videos in the first arrangement. Neither does in the second. Grouping by source changes the question you can answer about a new video. You would still need participant identities to test transfer to new people.

## Hide an observation and predict its description

Draw four columns for four moments in a walk. Draw three rows for the hip, knee, and ankle. Cross out the final ankle position because the camera missed it. Cover the knee in the second column with a scrap of paper. Which spot can give the student a target during masked training?

**Check your reasoning.** The covered knee was observed. The teacher can describe it, and the student can try to predict that description. The camera supplied no measurement for the crossed-out ankle, so there is no observed target there. The uncovered cells give the student context.

## Move one ankle and measure the forecast error

Suppose an ankle's last horizontal coordinate is 2 and its next observed coordinate is 5. A persistence forecast copies 2. A second forecast predicts 4. Square the difference between each prediction and the observed 5. Try another observation of 2 and compare the two errors again. These numbers are invented for practice.

**Check your reasoning.** At 5 the squared errors are 9 and 1. At 2 the errors are 0 and 4. The very same prediction can win or lose depending on how far the ankle actually moved. In the experiments, the mean squared error averages many observed coordinates.

## Compare four pairs of recordings

Two normal recordings receive scores of 1 and 4. Two abnormal recordings receive 3 and 5. A larger score is supposed to indicate more unusual gait. Pair each abnormal recording with each normal recording. How many of the four pairs are in the intended order? What happens if the abnormal score of 3 becomes 4?

**Check your reasoning.** The score 3 beats 1, and 5 beats both 1 and 4. That gives 3 of 4 pairs, or an AUROC of 0.75. With a score of 4, the tie with the normal 4 earns half a point, giving 3.5 of 4, or 0.875. This measures ordering. A rule that assigns a yes-or-no label would also need a threshold.
