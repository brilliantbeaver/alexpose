# Independent review of slides 31 through 43

Reviewed on 15 November 2026. The reviewer did not write the speaker notes for these positions. This review is read-only except for this report.

The reviewed [deck specification](../../deck_spec.json) has SHA-256 `965d8a236d3ab7b91253e009833d1e816dbd840d3382ed35819a6b74502c8ef4`. The reviewed [PDF](../../tutorial.pdf) has SHA-256 `b121a5100080869010838d42ac1c356f378cf31b02fa4cdbf64253e365225505`. These identify the version assessed here, rather than a later rebuild.

The review covers all 13 note entries and their visible text, individual renders 31 through 43, and contact sheets 04 and 05. Numerical checks used the frozen GAVD timing and classification records, the laboratory protocol and evaluation records, and the paired completion audit. I did not refit models or reinterpret a new estimator as an implemented result.

## Overall judgment

The sequence is accessible and scientifically restrained. The transition from category classification to the timing diagnostic, independent laboratory evaluation, measurement limits, and proposed work is coherent. The notes distinguish useful completion from anomaly-ranking improvement and preserve the positive completion findings. They do not claim a neurological mechanism, transferable GAVD weights, or a universally better mask. The inspected visuals have readable text, useful whitespace, and no observed text or line overlaps.

I found no incorrect displayed numerical result in these positions. Two attribution clarifications should be made before final release, and one speaker-note sentence should be aligned with its actual chart. These are reporting corrections, not evidence of a model or statistics bug.

## Material attribution clarifications

### 1. Slide 34 needs to identify Random U explicitly

The table reports `0.000393 [0.000177, 0.000618]` for the earlier edit and `0.000064 [-0.000232, 0.000368]` for the later edit. These are the outer-test normal results for **Random U**, with 27 recordings at each offset. Neither the visible slide nor its notes names that policy. This matters because slide 33 has just summarized all three policies across 24 combinations, so a listener could reasonably interpret slide 34 as an aggregate across them.

The values are correct. In `artifacts/iteration02/d66e1b126c9d9a9adf1b/timing_summary.json`, the matching records have `scope="outer_test_normal"`, `policy="U"`, and `variant="timing"`. Their unrounded means are `0.00039251818454954374` and `0.000064111437485884`.

**Concrete correction:** Add “Random U; 27 held-out normal recordings” to the lede or table caption, and begin the explanation with “This table uses the Random U model.” Retain the current warning about inadequate aligned prediction. No new numerical analysis is needed.

### 2. Slides 35 and 38 should state the laboratory score adaptation

The notes correctly describe separately fitted joint-angle models and explicitly reject video-weight transfer. However, the laboratory score discussion does not tell the listener that its movement descriptors and reference count differ from the GAVD definitions introduced earlier. Earlier slides define Movement M using 16 video features and Cross-leg R using five reserved normal recordings. The laboratory protocol uses **12 angle-level and angle-range descriptors** for M and **20 separately reserved healthy participants per fold** for R.

This is an omitted distinction, not an incorrect AUROC. Reusing the same symbols is reasonable because the score construction is analogous, but the changed measurements should be explicit enough that listeners do not carry the earlier feature set and reference count into the laboratory result.

**Concrete correction:** Add one sentence to slide 38's explanation: “Here, Movement M uses 12 joint-angle level and range descriptors, and Cross-leg R uses 20 reserved healthy reference participants per fold.” The visible slide can remain uncluttered.

## Speaker-note alignment correction

### 3. Slide 36 describes a plot that is not displayed

The visual paragraph begins “Read each fold across the methods,” then gives interpolation, baseline, own-correction and both-correction mean errors. The actual chart has one percentage-reduction bar per fold. It does not display four methods or absolute squared-degree errors.

The supplementary numbers and their ordering are verified by `external_fb9d0d23b56ce29fd846/derivatives/interpolation_audit_v1/summary.json`. They may remain useful supporting context, but the presenter should not direct the audience to read them from this chart.

**Concrete correction:** Begin the paragraph “Each bar shows the reduction in mean error when the Both-leg correction replaces the Own-leg correction in one test fold.” Introduce the four absolute errors as a separate saved comparison, using “The accompanying saved comparison also shows...” if they are retained. The displayed bars and values require no change.

## Verified scientific boundaries and results

| Positions | Verification |
| --- | --- |
| 31 | The five-class table correctly rounds mean macro F1 to 0.251, 0.309 and 0.347, and balanced accuracy to 0.255, 0.310 and 0.343. The notes separate normal-only encoder learning from label-trained classifiers and retain the 89-recording population. |
| 32 | The 48-sample parent, central 32-sample target interval, and offsets of four samples correspond to the implemented no-wrap construction. The visible and spoken restrictions retain fixed targets, own-leg inputs and body reference, and require valid permitted donor and attachment hips. Geometry and timing remain explicitly inseparable in this edit. |
| 33 | Both offsets retain 27 outer-test normal sources, with fold counts 7, 7, 6 and 7. Mean permitted distal-token counts for the Random U timing rows are 1.7467 and 1.7492, consistent with “about 1.75.” The summary contains 24 recipe/fold/offset adequacy decisions, all false, derived from 72 corresponding internal comparisons, all without improvement over both controls. These are related checks, not 96 independent studies. |
| 34 | The two means and intervals match the Random U timing rows exactly after rounding. Own-leg prediction changes are zero. The notes retain the inadequate aligned prediction, unmatched noise displacement and absence of a pure coordination interpretation. |
| 35 | The selected 188 trials and exclusion of ten healthy trials give 178 eligible participants: 128 healthy and 50 stroke. The notes distinguish processed laboratory angles from GAVD coordinates and describe separately fitted models on a common clock. |
| 36 | The four relative reductions are 5.5225%, 5.4171%, 5.0833% and 4.7757%, matching the bars. Every fold has 32 held-out healthy participants. The 72-column corrections have recorded ranks 44 and 62; the notes correctly limit capacity-free interpretations. |
| 37 | The saved paired audit gives mean errors 201.0377 and 190.5836, a 5.2001% relative reduction, interval [3.5530%, 6.6976%], and 109 of 128 participants improved. The absolute reduction and interval also match. The post-outcome, fixed-model, 1,000-resample status is explicit. |
| 38 | Laboratory weighted within-fold AUROCs round to 0.958, 0.694 and 0.892. Combined minus Movement is -0.06625 with interval [-0.105625, -0.030625]. The notes appropriately avoid claiming that all uses of opposite-leg information are ineffective. |
| 39 | The matching audit confirms 45 unreused same-sex pairs within five years and within test fold, split 11/12/11/11. The sex counts, mean absolute age difference, AUROCs and paired interval all match. The notes preserve the main population, unchanged models and unmatched walking speed. |
| 40 | The image-boundary counts and pooled fractions match the recorded audit. The notes distinguish source uploads from verified people, acknowledge exploratory use of all 28 normals, and do not turn three-frame inspections into tracking validation. |
| 41 | The 19.5%, 5.2% and 7.7% results retain their separate comparators and measurement systems. Completion, detection and mechanism claims remain distinct. |
| 42 | The notebook topics and `uv` commands match the current project README. Independence is framed as fresh-kernel execution with exact saved artifacts, not automatic regeneration of every model during an ordinary tutorial run. |
| 43 | Tighter capacity control, normal-data calibration, additional seeds and new participants are clearly proposed work. The slide does not promise that these changes repair detection or establish a clinical mechanism. |

## Optional polish

The slide 35 notes could call the ankle channel “dorsiflexion” when introducing the six signals, rather than applying “flexion” collectively to hip, knee and ankle. This would be slightly more anatomically precise without adding detail to the visual.

Slide 42's references currently link notebook 12 and the protocol. Direct links to notebooks 13 through 17 and the verified execution record would make its statement about linked notebooks easier to use. The existing project README already provides the notebook map, so this is navigation polish rather than a reproducibility blocker.

All transitions from positions 31 through 42 point to the next displayed topic, and position 43 closes the talk appropriately. No reordering is needed.
