**Role**: You are an expert world model (JEPA) researcher well versed in human gait analysis with world models

**Task**: You are to carefully and thoughtfully create a revised and improved version of nb_05a based on README_3_WEEK.md and METHODOLOGY_3_WEEK.md in `notes/ideas-claude-05-signed-laterality-decodability`.

The important shift is to stop treating the GAVD result as the clinical experiment. It is a code and representation audit; the stroke cohort supplies the decisive evidence.

Ultrathink on how to logically and systematically implement the following steps to enhance the scientific rigor and technical validity of the signed laterality experiment for the GAVD dataset. Do not add loaders or steps for the stroke or MoVi datasets yet, since that will come after ensuring that the approach works for the participants in the internal GAVD dataset.

## 1. Establish shared, tested protocol code first

Do not duplicate core logic across notebook cells. Move the frozen scientific rules into importable utilities/
configuration:

* common joint schema and per-dataset adapters;
* canonical coordinate-frame construction, pelvis centering, robust leg-length scaling, masks, and joint cycle resampling;
* anatomical reflection M: negate mediolateral coordinate and swap every semantic left/right joint, including masks/confidences;
* right-minus-left target convention;
* participant/source manifests, fixed split files, fixed label rankings, and seed schedule;
* Arms A–G, metrics, bootstrap, corruption generators, and result schema.

Add tests before running experiments:

* M(M(x)) == x;
* odd targets negate and even targets remain unchanged;
* reflection preserves sequence length, mask count, bone lengths, and forward direction;
* no person/source crosses a split;
* label-budget prefixes are nested and shared by every arm;
* exact Arm C output is odd to numerical tolerance.

This prevents the notebooks from becoming the only definition of the study.

## 2. Revise nb_05a into a strict GAVD audit

Keep it modest and honest.

1. Use the correct real cache explicitly. Require a configured artifact root rather than silently falling back to smoke data. Smoke mode should remain only as a quick wiring test and should watermark all plots/results as synthetic.

2. Make cohort selection explicit.
* Primary table: common canonical extraction path.
* Sensitivity table: wider local GAVD set, with extraction provenance included in the manifest and nuisance checks.
* Do not mix augmented-normal rows with canonical abnormal rows as though they were a homogeneous disease comparison.

3. Correct the sign convention. Preserve the historical stored target for reproducibility, but report:

[
y_{\mathrm{GAVD}}=-y_{\mathrm{historical}}
]

so every new result means “right greater than left” when positive.

4. Retain source-video grouping and label all outputs. Every result should say:
* transductive;
* source-video-grouped;
* signed coordinate excursion;
* hybrid JEPA checkpoint.

The frozen historical checkpoint saw these data and is not a clean inductive or clinical test.

5. Replace the old four-lane verdict threshold. The current “R² / raw-coordinate percentage / mirror-slope band” pass-fail rule should become descriptive pilot reporting:
* out-of-fold MAE and untruncated R²;
* raw-coordinate reference, random-encoder floor, side-agnostic and nuisance controls;
* mirror oddness error, slope, intercept, and sign-flip rate;
* target-permutation and left/right-shuffle falsification checks.

The raw-coordinate model is an expected strong reference, not a null, because the target is derived from the coordinates.

6. Separate reflection from viewpoint. Keep the anatomical mirror test, then add fixed rigid yaw rotations without swapping joint labels. A good laterality output should flip under reflection, not under a camera-frame change.

nb_05a should answer only: does this historically exposed representation retain a coordinate-derived signed signal, and does its output behave plausibly under reflection?

**Output** You should not replace the current nb_05a notebook. Instead, create a new version with your new changes and improvements.

Use adversarial review subagents to review and check all of your work, systematically and thoughtfully fix all issues.

Use fan out subagents with dynamic workflows to parallelize your tasks.
