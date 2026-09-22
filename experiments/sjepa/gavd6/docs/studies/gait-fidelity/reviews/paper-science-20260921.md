# Independent scientific review of the reorganized proposal

21 September 2026. Read-only review of the new proposal and data specification, performed independently of their authors. The reviewer also inspected the predecessor training code to distinguish inherited behavior from proposed implementation. This is a review of an experimental plan; it does not validate new results or data access.

**Files reviewed:** [proposal](../README.md), [data specification](../data/README.md), [evaluation protocol](../methods/evaluation.md), and the relevant [predecessor training implementation](../../../../src/gavd6_sjepa/research_directions/synthetic_training_v2/training.py). The evaluation protocol was still receiving its detailed percentile-endpoint contract during this review.

## Verdict

The reorganization gives the study a coherent scientific argument: a known movement change should remain measurable when observations are degraded, and improved coordinate error alone cannot establish that property. The explicit image-plane endpoint, separation of source evidence from model results, and dataset admission requirements substantially improve the proposal. It remains appropriately prospective, without presenting raw clinical-video availability as independent clinical validation.

One attribution issue should be corrected before the proposal is treated as the definitive experiment plan. Two measurement details must be explicit in the linked protocol before implementation. None requires a broader research program or another dataset search.

## Required correction: separate representation effects from trainable-stage effects

The initial draft applies the change constraint during frozen-encoder JEPA readout, while the corresponding direct model receives it during end-to-end training. Its comparison table then proposes direct and JEPA arms, each with and without the constraint. That comparison is useful for complete training recipes, but its interaction changes both the representation objective and the parameters allowed to adapt. It cannot by itself attribute an interaction to feature prediction.

Retain direct end-to-end training as the practical benchmark and add a stage-matched coordinate-pretrained encoder with the same frozen-readout protocol, both with and without the constraint. Apply the selected constraint and downstream labels identically to the initialized and shuffled-reference controls. Distinguish three comparisons:

| Contrast | What it establishes |
| --- | --- |
| Direct end-to-end with versus without paired-change supervision | The benefit of the constraint within the direct recipe |
| Paired JEPA versus coordinate pretraining, with the same frozen-readout stage and constraint | The contribution of the feature-pretraining recipe under matched downstream adaptation |
| Final candidate versus the strongest equally supervised direct recipe | Practical improvement over a strong deployable alternative |

Where an objective-by-constraint interaction is claimed, estimate it between stage-matched arms. Equal numbers of updates do not remove a difference in which weights can change. This correction also keeps a favorable recipe comparison from becoming an unsupported claim about a JEPA-specific mechanism.

## Protocol requirement: the same time support across paired cells

The README already requires common reference-defined frames across limbs and methods. Make explicit that a movement/observation contrast must also use the same physical interval and a declared common reference-supported time grid across its paired cells. Otherwise a percentile computed on different supported phases can change because the sample support changed. Camera-dependent references should retain the correct geometry while the eligibility contract is fixed for the contrast.

Reference-ineligible cells and source families must remain in coverage reports. If the complete four-cell contrast is unavailable, label that contrast unsupported rather than assembling it from unrelated intervals or silently replacing its population. A less restrictive partial-contrast analysis may be reported separately if declared in advance.

## Protocol requirement: missing predictions must affect the decision

The draft correctly says that missing or degenerate predictions count as failures. The implementation contract should specify how those failures affect the primary person-level decision, rather than merely retaining a failure-count column beside an error mean that drops missing values. Either use a justified prespecified failure score with sensitivity analysis, or require a prespecified coverage/failure criterion jointly with a clearly labelled conditional-error criterion. All methods must retain the same attempted reference-eligible population.

Keep this separate from reference ineligibility: an unmeasurable reference cannot supply a numerical error, while a missing output on a measurable reference is a model failure. The proposal need not choose arbitrary numerical penalties before development evidence exists, but it should require this decision before confirmation.

## Assessment of the percentile endpoint

`q = P95(theta) − P5(theta)` is a defensible initial projected angular-excursion definition. It reduces reliance on a single extreme frame without requiring contact events or additional landmarks. The proposal accurately explains that it discounts brief extremes and that its sign does not identify an affected limb. It also retains camera-specific references, anatomical side, original geometry, full waveforms and a full-range sensitivity analysis.

Its implementation must retain the following boundaries already stated or being added to the protocol:

- Compute the interior angle from hip–knee and ankle–knee vectors in original image geometry or after one shared isotropic transform. Anisotropic resizing changes the angle.
- Freeze the percentile interpolation convention, physical-time grid, interval duration, projected-segment threshold and completeness rule before comparing model performance.
- Use reference-defined support. A model cannot improve its score by making its hardest predictions invalid or nearly degenerate.
- Do not interpret agreement in excursion as agreement in phase, timing, waveform, anatomical three-dimensional range or clinical impairment.
- Compare the two limb values and level bias as well as their signed difference and within-source change.

The new constraint is properly labelled unimplemented. A quantile formula does not establish a stable differentiable objective: gradients may concentrate on the samples defining the percentile, and angle derivatives become troublesome near degenerate configurations. The proposed gradient checks are therefore necessary. Any surrogate must be declared, tested against the locked evaluation endpoint, and supplied identically to its matched controls.

## Data and inference boundaries that pass review

The data specification is concrete about what currently exists and what is pending. It correctly distinguishes local raw videos from future independent annotations, filename-derived sources from verified people, and recorded pathological source motion from synchronized real-image clinical references. It also states that local AMASS raw assets are absent and must be verified on HAIC rather than inferred from result downloads.

The local MS/PD/Normal footage remains development material; the GAVD overlap and cache defects are explicit. Artificial occlusion of originally annotated visible joints is a bounded real-image recovery experiment. It does not supply reference truth for joints hidden in the original recording. The clinical candidates remain conditional on access, alignment, independent references and sample-size feasibility.

The main clinical boundary is sound: better measurement of side-specific change could support assessment, whereas rehabilitation benefit requires separate clinical evidence. The study also preserves the distinction between physical movement change, camera projection and estimated joint renaming. Mirror controls are not presented as a graded disease simulation.

## Remaining implementation gates

The finalized proposal should retain, rather than conceal, the unresolved implementation requirements: a validated graded-motion panel; reference support for the new endpoint; a frozen failure policy and test population; the body12 graph adapter; source-group identity checks; the paired-change loss; and synchronized input/reference/output review displays. These are appropriate gates in a prospective proposal and must not be described as completed experiments.

The reviewer did not inspect final rendered figures or browser interactions for this scientific review. Their visual and functional acceptance requires a separate review record.

## Final disposition after revision

The reviewer independently re-read the changed proposal, methods index, primary-endpoint contract and literature attribution matrix on 21 September 2026. All three requested scientific corrections are now addressed:

1. Coordinate-pretrained and JEPA encoders share the frozen-readout stage, with and without the paired-change term. Initialized and shuffled-reference controls receive the same selected readout objective. Direct end-to-end comparisons remain explicitly separate, and the literature matrix now reflects this distinction.
2. The full two-by-two comparison and interaction explicitly use the intersection of reference-supported physical timestamps across all four cells, both limbs and methods. Unsupported complete contrasts remain in coverage accounting.
3. The primary decision must use either a declared all-attempted failure score or a joint coverage-and-conditional-error rule with explicit confidence criteria. An error mean that drops failed predictions cannot determine success by itself.

The proposal and protocols also consistently identify the inherited EMA reference teacher, the frozen coordinate-readout stage, and the absence of the proposed new loss from the predecessor implementation. Linear percentile interpolation is now specified for the initial uniform-time synthetic calculation. Interval length, reference support thresholds, sign-resolution tolerance, margins and failure penalties remain to be fixed during reference-only admission. Those pending numerical decisions are appropriate for a proposal and must be frozen before model comparison or confirmation as specified.

**Scientific disposition: accepted as a prospective research proposal with explicit implementation and data-admission gates.** This acceptance does not certify an implemented method, successful training, statistical significance, clinical validity, final figure layout or browser behavior.
