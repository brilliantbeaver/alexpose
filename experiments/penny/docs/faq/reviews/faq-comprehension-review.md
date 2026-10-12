# FAQ comprehension and conceptual review

Reviewed source: `docs/faq.md`, version inspected on 4 October 2026. Line numbers below refer to that version and may move during revision. This review reads the document as a novice trying to connect the learning objective, measurement choices and retained results. It did not independently recalculate the result tables or inspect the rendered Word document.

## Issues to address

### C01 — Correct the retained coordinate frame

**Location:** Question 17, approximately lines 164–166; Question 36, line 312.

**Issue:** The phrase “crop-relative coordinates” implies that the cached x/y values use the moving crop as their origin. Notebook 02's `detect_pose_in_crop` explicitly transforms detected crop coordinates back to normalized full-frame coordinates: `(x0 + landmark.x * crop_width) / width`, and the analogous y expression. This matters because a reader could incorrectly conclude that crop movement is directly counted as pelvis movement by the later proxy calculation.

**Suggested replacement in Question 17:** “The detector runs on the crop, and its x/y landmarks are mapped back to normalized coordinates in the original frame. These image coordinates and model-estimated relative depth are not calibrated distances in meters.” Retain the following explanation of camera, perspective and occlusion effects.

**Suggested replacement in Question 36:** “Their interpretation depends on camera motion and perspective, while crop quality can also change the detector's estimates.” This preserves the genuine crop-related limitation without misdescribing the coordinate origin.

### C02 — Define the working null in terms of measurable improvement

**Location:** Question 3, line 45.

**Issue:** “Adds no demonstrated ... information beyond the information already available in the input” mixes a hypothesis about an effect with the current state of its demonstration. Any deterministic encoder derives its features from its input; the actual experimental question concerns whether pretraining organizes that information so that specific quantities or labels are more recoverable. “No demonstrated” also cannot become a prospective test criterion without identifying the comparison and metric.

**Suggested replacement:** “For this FAQ, the starting working null is that self-supervised pretraining does not improve recovery of specified gait quantities or prediction on independent recordings compared with an untrained encoder and simpler input summaries.” Continue with the existing explicit qualification that this is retrospective and has no preregistered decision rule.

Question 4 already supplies the narrower controls. This change makes the story's starting question match those tests without claiming that the current corpus can resolve independent-recording performance.

### C03 — Explain the Random Forest before reporting its results

**Location:** First use in Question 12, line 119, and Question 15, line 141.

**Issue:** The document explains encoders and linear probes but never defines the classifier that supplies all five-class and binary headline scores. “Nonlinear classifier evaluation” presumes the reader knows what a Random Forest does and why it differs from a linear readout.

**Suggested addition to Question 15:** “A Random Forest combines many decision trees. Each tree asks a series of questions about feature values; their category estimates are averaged to choose the predicted category. The combined decision boundaries can be more complicated than a single linear readout.” The earlier mention can point forward to Question 15.

For the next paragraph, replace “a comparatively simple linear readout” with “a weighted sum of the input feature values, fitted to predict the target.” Keep the explanation of regularization and the limits of a failed probe. These additions are basic definitions rather than extra implementation detail.

### C04 — Show why full-clip preprocessing compromises the forecast boundary

**Location:** Questions 40–41, lines 355–365.

**Issue:** The document names the correct limitation but gives no example that explains how future data can affect an earlier input. A novice may assume that removing the future tokens is sufficient and treat the preprocessing qualification as a generic disclaimer. The future-target teacher is also described only in the earlier architecture section, leaving room to misread the future experiment as predicting independently measured future coordinates.

**Suggested addition to Question 40:** “For example, body width is summarized over the entire clip before normalization, so changing the unseen later frames can change the normalized earlier frames. The target encoder also receives the complete clip, including the future, and supplies the latent targets against which forecasts are scored.”

This path is confirmed by the deterministic preprocessing check recorded in `historical-evidence.md`: identical raw 48-frame prefixes with different future shoulder widths produced different prepared prefixes. The FAQ need not quote that fixture's numeric difference, which is an implementation check rather than a research result.

**Suggested further sentence:** “The four future segments are predicted together; the notebook does not feed one predicted segment back in to generate a longer rollout.” This prevents “world model” and “rollout” language elsewhere from implying an experiment that was not performed.

### C05 — Distinguish standard cadence from the notebook's proxy

**Location:** Question 36, line 310.

**Issue:** “Cadence describes movement repetition per unit time” is too broad for a basic gait definition and leaves unclear why the following cycles-per-second value needs qualification. In gait, the standard term counts steps; the notebook's signal counts ankle oscillations. Their relationship depends on how the oscillation cycle is defined and detected.

**Suggested replacement:** “In gait analysis, cadence is the number of steps per unit time, commonly reported as steps per minute. Notebook 08 instead estimates a left-ankle oscillation rate in cycles per second. Converting that signal into step cadence requires checking what one detected cycle represents.”

Keep the camera and proxy limitations. In Question 37, add one concrete reliability statement: “Weak probe scores may also reflect noisy targets: short clips, long detection gaps and walking toward the camera can make an ankle-derived timing signal unreliable.” This addresses the inference that weak readout alone proves failure to encode a well-measured physical quantity.

### C06 — Expand the anatomical terminology where it first matters

**Location:** Question 20, lines 186–188.

**Issue:** “Bilateral,” “flexion” and “proximal weakness” are anatomical jargon in the answer intended to explain the landmark motivation. The reader can follow the model machinery but still fail to understand the body-level hypotheses linking the four conditions to the selected regions.

**Suggested replacement:** “The project maps reviewed gait features to both shoulders, hips, knees, ankles and foot-index landmarks...” In the next paragraph use “increased bending of the lower-limb joints” for “lower-limb flexion patterns,” and “weakness near the hips and trunk, with compensating body movement” for “proximal weakness and compensatory motion.”

The mapping files contain aliases and proxies worth one concrete example: “For example, a stored field called step-length variability uses an ankle-separation proxy in the legacy feature pipeline, so its name alone does not establish a true step-length measurement.” This example should remain explicitly attached to the mapping/legacy pipeline; it must not suggest that the current learned encoder computes that field.

### C07 — Make macro F1 understandable without a mathematical vocabulary prerequisite

**Location:** Question 26, line 236.

**Issue:** “Harmonic mean” is unexplained mathematical jargon inside the requested basic metric explanation. The definition is technically correct, but the reader cannot infer why this metric is useful or how a classifier can have decent accuracy with a poor category balance.

**Suggested replacement:** “F1 combines precision and recall as `2 × precision × recall / (precision + recall)`, so a low value of either lowers the score. Macro F1 gives every category equal weight by averaging its F1 score.” A sentence can explain that predicting the large category well does not compensate for missing the small categories under macro averaging.

This is a short formula with a direct interpretation, not a demand for extra metrics or testing.

## Inference boundaries to preserve during revision

The following boundaries are consequential and should survive any shortening:

- The null is retrospective; neither the notebook order nor the FAQ's logical story should be presented as preregistration or a recorded chronological discovery process.
- Video-ID recovery does not establish that pretraining added identity information, since the untrained encoder is at least comparable on the retained splits. It also does not directly measure how much the condition classifier uses that information.
- The 24% grouped result is a stress test with missing class support and an exposed encoder, rather than a clinical generalization estimate. A 38- or 42-point gap is not a causal estimate of memorization.
- The anatomy-mask difference is one prediction under one seed; it supplies no demonstrated advantage, rather than proving equivalent performance for all training runs.
- The future comparison is in-sample model error versus a query-excluding, joint-averaged baseline. Future targets come from a full-context learned teacher, and full-clip preprocessing can carry future information into the input prefix.
- The phase result is a failed readout of a signal-derived timing proxy. The recorded “circular_correlation” field should continue to be named mean cosine phase alignment.
- Abnormal forecast error combines training familiarity with condition and recording changes. No health-condition detection threshold or independent normal performance has been established.

## Scope and style disposition

No document-wide rewrite is needed to implement these findings. The most useful revisions are local corrections and examples at the point of first use. Preserve the question order and the explicit links between observed result, alternative explanation and next experiment. Avoid adding a repeated “this is not X” disclaimer to each definition; the existing protocol-specific qualifications can carry the necessary inference limits.
