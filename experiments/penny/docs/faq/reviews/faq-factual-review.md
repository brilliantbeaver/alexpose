# Factual review of the illustrated FAQ

Reviewed `docs/faq.md` and the SVGs it references against GAVD3 notebook code and retained real artifacts on 2026-10-04. Main draft SHA-256 at initial read: `2f06e687f5cf4e3167289ac7c205af55ebb4c9b1befe8dc73790dc566e273120`. This is a read-only factual review of the main document; it is separate from the CLI adversarial review. No model or pose experiment was rerun.

Verdict: needs-attention. The retained numbers and the principal evaluation caveats are accurate. The required corrections concern input-coordinate definitions, phase-reader methods, and figure descriptions. Delivery links also need a final check after the document build.

## Issues requiring revision

### FR-01 — Medium: cached coordinates are full-frame normalized, not crop-relative

**Location:** FAQ Question 17, line 166; Question 36, line 312.

**Evidence:** Notebook 02, `detect_pose_in_crop`, remaps estimates after crop inference:

```python
sequence_row[index, 0] = (x0 + landmark.x * crop_width) / width
sequence_row[index, 1] = (y0 + landmark.y * crop_height) / height
sequence_row[index, 2] = landmark.z * crop_width / width
```

The pose is estimated from the crop, but downstream x/y are expressed relative to the full source frame. The current sentence “crop-relative coordinates are not distances in meters” misstates the actual stored coordinate system. Moving a crop does not itself recenter the stored x/y because its offset is added back.

**Required correction:** State that crop-based estimates are mapped back to full-frame normalized x/y before caching, with relative estimated z. Explain that the resulting pelvis motion is camera/image-relative, not calibrated ground speed. A changing crop can affect detection quality, but should not be described as the coordinate frame of the cached trajectory.

### FR-02 — Medium: explain how image normalization affects angle and displacement proxies

**Location:** Questions 36–37, especially line 312; potentially Question 17.

**Evidence:** Notebook 08 `_knee_angle_series` computes Euclidean vector dot products and norms directly on cached x/y. Notebook 02 normalizes x by source-frame width and y by source-frame height. Unless aspect ratio is corrected, equal normalized coordinate differences along x and y do not correspond to equal pixel distances. The knee proxy is therefore an angle in a normalized image plane; source aspect ratio and camera projection influence it. Its excursion is the 95th-minus-5th percentile of that angle, rather than its maximum-minus-minimum range. `speed_proxy_normalized_per_second` computes framewise pelvis displacement in this same image coordinate system and multiplies by FPS; `trunk_sway_normalized_units` measures detrended pelvis x variation.

**Required correction:** Add a short concrete explanation: image displacement measures the pelvis moving across the source frame and can include camera motion; knee excursion is a percentile span of a two-dimensional pose angle, affected by perspective and uncorrected x/y aspect scaling. Keep the numbers and units labeled as proxies. This avoids teaching that model-estimated values are physical walking speed or laboratory joint angles.

### FR-03 — Medium: the phase test uses segment features and a different split from scalar probes

**Location:** Question 37, line 327, and the preceding method explanation.

**Evidence:** Notebook 08 `frame_latents` computes mean joint-token vectors per temporal segment, then repeats each segment's vector across its four frames. `grouped_ridge_oof` uses five-fold `GroupKFold` with whole sequence IDs as groups. Scalar proxies instead use the pooled 384-number sequence vector with leave-one-sequence-out readouts. The draft gives the phase R² but does not explain this method change, and subsequently refers to “frozen pooled features” when interpreting the combined results.

**Required correction:** Say the phase readout receives a separate 96-value segment representation, repeated over its four frames, and keeps all frames of a sequence in one of five evaluation folds. This prevents nearby frames appearing on both sides but still allows source-video overlap. Its negative score concerns that temporal readout; it should not be presented as another scalar probe of the 384-number sequence pool.

### FR-04 — Medium: Figure 10 claims nonexistent error bars

**Location:** Figure 10 caption, line 294; `docs/faq/figures/identity-probe.svg`.

**Evidence:** The SVG contains four mean bars and one dashed uniform-guess reference. Its line elements draw the title divider and dashed reference; there are no standard-deviation whiskers. The caption currently says “Error bars show standard deviation across three repeated splits.”

**Required correction:** Either add accurately scaled standard-deviation whiskers from `07_video_identity_probe.csv` (0.121099, 0.158020, 0.039817, 0.019909), or change the caption to say bars show mean accuracy and the spread is reported in the adjacent text. Do not call the bars uncertainty intervals.

### FR-05 — Medium: Figure 11 says the raw-coordinate control is shown, but omits it

**Location:** Figure 11 alternative text, line 329; `docs/faq/figures/gait-probe-results.svg`.

**Evidence:** The SVG has columns for S-JEPA and missingness only; the alternative text says “raw-coordinate and missingness controls shown beside them.” The main Markdown table does include raw-coordinate R², so the numerical record is otherwise complete.

**Required correction:** Add the raw-coordinate column or align the alternative text and caption with the two feature families actually displayed, explicitly pointing readers to the full table for the omitted comparison.

### FR-06 — Low: Figure 5 descriptions do not describe the displayed illustration

**Location:** Figure 5 alternative text and caption, lines 190–192; `docs/faq/figures/condition-proxies.svg`.

**Evidence:** The figure depicts condition annotation branching into movement and recording/extraction pathways, then features. It does not depict bilateral regions or connect individual anatomical regions to gait quantities, as the alternative text and caption imply.

**Required correction:** Either supply the described anatomical diagram or rewrite both descriptions around the actual competing pathways. The figure's current association caveat is appropriate and should remain.

### FR-07 — Low: distinct upload identifiers do not establish independent recordings or participants

**Location:** Notebook 01 description in Question 5, line 57 (“count independent source videos”).

**Evidence:** The source audit counts unique `video_id` values. FAQ Question 30 correctly acknowledges that different videos might contain the same person. The retained metadata cannot prove separate participants, separate sessions, or absence of reuploads.

**Required correction:** Use “count distinct source videos” or “preserve and count source identifiers.” Continue to describe source grouping as a minimum separation unit, with participant independence requiring more metadata.

### FR-08 — Delivery gate: four local targets absent at first review

**Location:** Links from Questions 16–17, the download link, and reproducibility notes.

**Evidence:** The initial link scan checked 49 local targets and found these missing:

- `docs/faq.docx`
- `docs/faq/evidence/original/source_video_census.csv`
- `docs/faq/figures/pose-pipeline.svg`
- `docs/faq/reviews/revision-record.md`

Three may be planned build/review outputs rather than content defects. `source_video_census.csv` was not among the initial copied evidence files; the retained `source_video_concentration.csv` already supports the counts and is an available alternative target.

**Required correction:** Create the planned outputs before delivery and either preserve the source census with its provenance or link to the existing concentration table. Rescan local paths after regenerating Markdown/Word. Recompute provenance if any evidence file is added or changed.

## Numerical verification

All numerical research values printed in the reviewed Markdown match their cited artifacts at the displayed precision:

| Claim group | Verification |
| --- | --- |
| Cohort | 96 sequences, 18 video IDs; normal 12/1, Parkinson's 9/2, stroke 12/3, cerebral palsy 16/2, myopathic 47/10. |
| Architecture and masking | 64 frames / four frames per patch = 16 segments; 16×33 = 528 tokens; all-valid eligible pool 16×10 = 160; 0.60×160 = 96 hidden, 96/528≈18.2%; 96 channels / four summaries = 384; 700,800 trainable parameters. |
| Pretraining | 12 normal inputs, 300 epochs, batch four; EMA start 0.999; target/predictor temperatures 0.06/0.10. Training history endpoints 12.54→0.57, standard deviation 0.340→0.412, cosine 0.636→0.535. |
| Exact classification | 47/21 split; S-JEPA 13/21=61.9%, balanced accuracy 59.6%, macro F1 0.613; saved handcrafted 16/21=76.2%, macro F1 0.728; three predictions / 21 = 14.3 percentage points. Majority and missingness each 7/21=33.3%, missingness macro F1 0.336. |
| Exploratory classification | 67/29 split; S-JEPA 18/29=62.1%, macro F1 0.594; majority 14/29=48.3%; missingness 13/29=44.8%; binary tests seven to 18 inputs and AUC 1.0 for stroke/cerebral palsy. |
| Exact errors | Myopathic 5/7; cerebral palsy 3/5; normal 2/3; Parkinson's 2/3; stroke 1/3 with two cerebral-palsy predictions. One prediction / 21 ≈ 4.8 points. |
| Exposure | All nine exact and 16 exploratory test video IDs present in classifier training; three exact and four exploratory normal test sequences present in encoder pretraining. |
| Identity probe | Three split seeds: means 49.4%, 51.7%, 46.0%, 42.5%; standard deviations 12.1, 15.8, 4.0, 2.0 points; nominal uniform reference 1/18≈5.6%. |
| Grouping diagnostic | Six folds ×16 examples; 23/96≈24.0%, fold standard deviation 16.5 points; training-majority 47/96≈49.0%. Sequence-fold comparator 63/96≈65.6%; gaps 38.1 and 41.7 points. |
| Scalar and phase probes | All 18 scalar R² table entries agree with `08_gait_parameter_probe_scores.csv`; n=96 except stance/swing n=95; mean cos/sin phase R² −0.127. |
| Masks | One real seed; neurologic/motion-aware 12/21≈57.1%; random/full-body 13/21≈61.9%; all prepared-clip cadence R² negative. |
| Forecast | Context 48 prepared frames, suffix 16; four horizon losses 0.353,0.354,0.341,0.376; coarse baseline range12.12–12.60; modelphase1.531rad=87.7°, persist1.372rad=78.6°; mean cosine alignment0.030 vs0.146. |
| Condition forecast means | Normal0.362, Parkinson's2.541, stroke3.113, cerebral palsy2.965, myopathic3.038; normal reference uses training clips. |

All 41 evidence files listed in the initial provenance manifest exist and match their recorded SHA-256. External reference pages were not re-fetched by this factual review; the parent review should retain its source-verification record.

## Claims that hold, with stated boundaries

The draft appropriately describes the null as a retrospective organizing hypothesis, avoids declaring statistical rejection, and treats the one-seed mask finding as no demonstrated advantage rather than equivalence. It correctly distinguishes source identity being available from a quantified causal contribution to label classification. The grouped diagnostic discloses missing normal classifier training support and pretrained normal exposure; the forecast section discloses full-clip preprocessing, an in-sample normal reference, a coarsely pooled comparator, and unvalidated abnormal-error specificity.

Accuracy, class recall/precision, balanced accuracy, macro F1, ROC AUC, and R² definitions are accurate for the stated protocols. The phase alignment statistic is correctly renamed in prose rather than perpetuating the artifact's misleading `circular_correlation` label. None of the checked numbers requires correction.

Remaining explanatory caveats should focus on the coordinate system and distinct scalar/phase representations. No new significance tests, training runs, or clinical claims are warranted by this review.
