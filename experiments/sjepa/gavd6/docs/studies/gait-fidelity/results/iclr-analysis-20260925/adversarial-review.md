# Independent review of the gait-fidelity results interpretation

Three independent agents reviewed the downloaded evidence before the synthesis
was written, then challenged the assembled draft and figures. Each worked from
the actual files in `outputs/iclr` and relevant study definitions. They did not
change the experiment files or select a replacement primary endpoint.

| Review area | Independent checks | Changes incorporated in the final writeup |
|---|---|---|
| Statistics | Reconstructed all saved core/response effects and bootstrap intervals; checked repair person-level contrasts, three-seed uncertainty, zero-response baseline, failure decomposition, multiplicity, and figure intervals | Kept all three primary comparisons unresolved; retained favorable endpoint repair as secondary; distinguished person-balanced direction accuracy from a raw pooled fraction; corrected the distinction between an additive contribution and a success-conditional mean |
| Representation and repair mechanisms | Read all 21 representation diagnostics and six calibrations; checked loss definitions against recorded source hashes; inspected frozen encoder identities and training budgets | Attributed the positive probe flag to the masked reference-input teacher; separated deployment encoder and teacher probes; limited collapse claims; explained shared geometry loss and scalar-weight intervention; clarified endpoint versus delta auxiliary losses, gradient definition, and clipping |
| Population and provenance | Verified all 69 SHA-256 hashes and frozen configuration/plan identities; audited train/development disjointness, actual versus planned counts, source composition, reused metrics, diagnostic folds, and held conditions | Used 112 training and 14 development people; preserved repeated-observation counts; identified the 333-pair diagnostic panel as covering 112 training people; kept ViTPose repair separate from pooled results; clarified that direction eligibility uses absolute reference-change magnitude |

The initial reviews ruled out several tempting interpretations. A mean delta-JEPA
advantage does not establish a primary effect, the experiments are not independent
replications, and lowering the scalar weight already achieves most of the mean
waveform recovery attributed to dense supervision. Approximately 74% of the
response primary's mean difference is in the failure contribution. The positive
teacher probe does not establish successful deployment-encoder probing. Nonzero
feature variance does not rule out partial collapse.

The final review found no numerical discrepancy in the report's main tables or
figures. It requested wording corrections rather than a changed scientific
conclusion. The report now describes gradients as derivatives of the loss with
respect to parameters and distinguishes clipping those gradients from rescaling
optimizer updates. Its synthesis states the observed loss-weight intervention
result directly instead of claiming a complete causal explanation.

A final statistical review independently checked the condition summaries. The
nonheld response stratum contains two intervention levels and the held stratum
one, requiring 2:1 weights; endpoint waveform and coordinate strata require 4:1
weights because baseline and its no-change copy are also present. Recombining
the resulting observation strata reproduces all 16 neural methods' published
means to numerical precision. The report describes clear observations as lacking
added occlusion, while retaining the other corruptions in that condition panel.

Figure 2 was independently checked: the blue intervals are crossed person/seed
bootstraps, and the orange repair interval is the declared person t interval.
Figure 3 retains both repair controls and the direct benchmark. Figure 4 includes
all failure costs; its blue segments alone assign zero contribution to failures,
whereas a conditional mean excludes failed cases. Figure 5 separates reference-input
teacher features from observed-input encoder features and labels squared-degree
units. All five PNG figures were also visually inspected for readable labels,
legends, axes, and clipping.

The reproducibility script rechecks the 69 input hashes, the common 14-person and
three-seed panel, exact imported-method agreement, 16 saved metric comparisons
across the declared contrasts, and additive error decompositions. These 16 checks
include secondary metrics; there are three declared primary endpoints. It writes
the figures, derived numeric tables, and [verification record](verification.json).

Remaining limits are substantive: no protected-person confirmation, no GAVD or
clinical evaluation, only three fitted seeds, and reuse of the development people
across adaptive follow-ups. Raw videos, checkpoint inference, full feature arrays,
and optimizer histories were not downloaded, so this review verifies the compact
evidence and its interpretation without rerunning the source experiments.
