# Submission readiness

The manuscript presents a complete methodological study of controlled cross-leg prediction for gait anomaly detection. Its strongest empirical evidence is the completion benefit on a fixed video-pose diagnostic and on independent laboratory participants. Its contribution to anomaly detection is a procedure that tests prediction, observation support and score integration separately.

## Scientific strength

The independent participant experiment is particularly useful: healthy completion error falls by 5.2%, 109 of 128 participants improve, and all four fold means favor the both-leg predictor over the own-leg correction. The GAVD experiment adds heterogeneous video observations and an equal-budget masking comparison. Anatomical training helps the focused task, while the unchanged uniform task bounds the generality of that result.

The methodological distinction is concrete. Both prediction branches use the same observable answer, retain a local solution and obey explicit information restrictions. The timing experiment checks prediction quality again after the available inputs change. This supports a substantive measurement contribution to anomaly research. It does not establish a new mask family, a neurological theory or improved detection accuracy.

The measured anomaly rankings belong in the paper because they test the proposed use of the prediction signal. Movement-only and combined AUROCs are 0.694 and 0.576 in the four-condition GAVD comparison, and 0.958 and 0.892 in the laboratory data. The practical implication is that completion benefit alone does not determine how relational evidence should be calibrated or combined with local movement.

## Remaining scientific limitations

The main unresolved issue for the completion interpretation is unequal modeling flexibility. Using the same number of correction features does not make the models equally flexible. A stronger set of own-leg features could narrow the both-leg advantage. Other limits are the small development-informed GAVD normal population, estimated pose targets, shared body references, artificial withholding, one laboratory trial per participant and conditional-only uncertainty.

The timing diagnostic cannot support a coordination interpretation because the predictor fails its accuracy requirement on the inputs actually available to the comparison. The independent participant study covers healthy and stroke walking in a different measurement system. Neither dataset establishes clinical deployment or participant-independent generalization across the four GAVD conditions.

The highest-value extension is to match model flexibility more closely and estimate score scales using only normal development data, then evaluate new participants or repeated trials. That experiment could test whether dependable relational evidence adds detection value. It has not been executed in the present study.

## Reproducibility and presentation

The [evidence record](evidence.json) and [completion audit](advisory/completion_audit_v1/README.md) identify the actual fitted studies and saved predictions. The experiment package records 18 independently executed notebooks and 137 passing tests. These records concern the underlying experiments; rebuilding the manuscript is an editorial execution, not a new model run.

The [validation record](final_validation.json) gives the current PDF hash, pagination, font, citation, anonymity, link and numerical-source checks. The [visual review](visual_review.md) records page inspection. The manuscript uses anonymous AAAI-27 formatting. Formatting and reproducibility can be verified mechanically; scientific significance remains a reviewer judgment.

The [review responses](response_to_reviews.md) record accepted corrections and remaining boundaries. The manuscript is suitable for review as an exploratory method and measurement study. It should not claim state-of-the-art detection, recovered disease mechanisms, clinical readiness or a guaranteed publication outcome.

The paper explains the method for general AI/ML readers. Four central illustrations appear in the PDF, while the [technical notes](technical_details.md) contain exact settings and two supporting plots. The [reader review](advisory/accessibility_review.md) checks that simpler language preserves the scientific distinctions.
