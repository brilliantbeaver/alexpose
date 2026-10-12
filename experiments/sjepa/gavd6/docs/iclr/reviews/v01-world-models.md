# Version 01: independent world-models and methods review

Reviewed `paper-v01.tex` and the diagram sources/PNG exports as a skeptical predictive-representation researcher. This reviewer did not author this manuscript. Numerical study outcomes receive a separate independent evidence audit. Findings refer to the frozen version as reviewed; later correction does not retroactively raise its score.

## Judgment

The paper has a coherent, appropriately bounded empirical claim: reference-feature prediction is evaluated by a concrete side-sensitive movement response, and three connected experiments expose an output-objective tradeoff without establishing a JEPA advantage. The prose openly rejects forecasting, planning, equivariance, clinical validity, and a general JEPA ranking. Its strongest evidence concerns the tested loss package and the importance of reliable measurement, rather than a new representation method. An ICLR reviewer may still find the small adaptive development population and fixed optimization sweep insufficient to establish a broad representation-learning result. That is a limitation requiring experiments, not stronger rhetoric.

## Substantive objections and dispositions

| ID | Severity | Objection and evidence | Required correction or reasoned disposition | Residual limitation |
|---|---|---|---|---|
| W01-1 | Material factual scope | Figure 1 caption says actual physical reflection plus anatomical exchange swaps image-plane excursions and reverses `A`. The implementation mirrors in 3D then projects through a fixed camera (`preparation.py:118–132,480–491`), which need not exactly exchange projected angular excursions. The numeric diagram itself presents that exact property as a physical contract. | Label the diagram as an **idealized algebraic side exchange**, and state that actual physical mirror variants are reprojected and remeasured per camera; do not impose `A_mirror=−A_original` on the study. | The synthetic task still does not establish learned equivariance or resolve natural anatomical ambiguity. |
| W01-2 | Moderate claim attribution | Abstract says “original scalar-change objective”; Results subsection says “scalar supervision damages trajectories.” The completed original intervention adds scalar and geometry losses together (`measurements.py:163–178`; `training.py:638–650`). Body text acknowledges this correctly, but headlines overassign the cause. | Use “original change-supervised objective” or “original change-supervised package” consistently. Attribute specifically to weight reduction only in the repair contrast that holds geometry fixed. | Dense versus scalar changes information content and gradient distribution; no unique mechanism is established. |
| W01-3 | Moderate methods | Dense equation gives a within-pair frame/leg mean but does not state the outer equal-pair reduction. Actual loss gives each eligible pair equal weight regardless of support count (`repair_objectives.py:20–37`). | Add an outer expectation over pairs or explicitly call the equation the loss for one pair and state the equal-pair mean. | No concern remains after explicit reduction. |
| W01-4 | Moderate reproducibility | “Centered and scaled from retained observations” leaves the critical leakage boundary implicit. Pretraining excludes artificial hidden coordinates before median/quantile normalization, which the teacher then shares (`training.py:288–321,569,600–608`). | Add the input-only median/span formula or a precise short description, including exclusion of hidden coordinates and fixed degenerate fallback. Identify confidence/availability separately from the artificial query mask in Figure 2. | No end-to-end reproducibility from the compact packet alone; raw assets/checkpoints remain elsewhere. |
| W01-5 | Moderate figure quality | Visual inspection finds the top arrow label in Figure 1 overlapping both boxes, left box borders clipped in Figure 2, and the residual arrow running through its own bottom explanatory label. | Increase gaps or split arrow labels onto short lines, inset boxes, and separate the residual arrow from its caption. Inspect at the final column/page width. | Small typesetting cannot supply a verified real reconstruction example absent from evidence. |
| W01-6 | Minor reproducibility | Base JEPA text names cross-entropy and a variance/covariance regularizer but does not specify its 0.05 weight or its invariance component. Repair calibration is verbal without the coefficient equation. | If space permits, state CE + `0.05×VICReg` (invariance, variance and covariance), and give `lambda_dense=0.1 sqrt(sum||grad Ls||²/sum||grad Ld||²)` over 32 initial training batches/readout parameters. | The schedule remains one fixed recipe; formula clarity cannot establish convergence. |
| W01-7 | Bounded research limitation | Paired finite-difference auxiliaries, centered teacher prediction, and dense supervision have precedents. The paper's contribution is diagnostic evaluation, not a new generic objective. Small adaptive development evaluation, retained reference teacher, and no tuning/convergence study limit generalization. | Retain the present explicit empirical-contribution positioning. Do not claim novel JEPA architecture or learned world dynamics. | New independent confirmation, stronger optimization controls and natural-video references needed for broader claims. |

## Confirmed correct and valuable

- Twelve joints and 384 four-frame tokens, four encoder blocks, two predictor blocks, EMA teacher, frozen residual readout and direct end-to-end comparator match executed configuration and implementation.
- Teacher input is projected reference; it is not a second observed-video view. Inference receives neither teacher, predictor, intervention metadata nor references.
- `e_i=H[p_i/0.1−sg((t_i−c)/0.06)]` and delta/endpoint auxiliary expressions match `response_objectives.py:38–57`; the coupling identity is stated only on identical tensors.
- Physical edits precede side-exchanged mirroring; naming corruption permutes coordinates/confidence/availability, with references fixed.
- The 15° intervention and ViTPose estimator are held out from fitting. Confirmation and GAVD are correctly described as absent.
- Core counts 30/9, response 18/9, repair 12/0 match completed work; unexecuted mask comparisons are disclosed.
- Base versus original paired objective is correctly identified as compound in the methods and figure caption. Repair retains identical geometry and calibrates to the low-scalar auxiliary, not coordinate gradient.
- Full-window restoration, pairwise movement differences and image-plane geometry are correctly distinguished from forecasting, temporal derivatives and anatomical joint range.
- The declared primary intervals remain unresolved, teacher probes remain privileged, and zero-response/failure accounting are not buried.

## Consistent rubric

| Dimension | Weight | Score /10 | Concrete reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution |20%|6.5|Clear structured representation evaluation; novelty is empirical and scope narrow, with no independent confirmation.|
| Claim accuracy and evidence support |20%|8.0|Strong boundaries and source-correct objectives overall; physical-mirror claim and scalar headline attribution need repair.|
| Evaluation and statistical rigor |15%|7.0|Paired hierarchical analysis and failure retention; only 14 adaptively reused people and three seeds.|
| Scientific insight and positioning |15%|7.5|Endpoint-control identity and readout-weight argument are informative; unique mechanism and generic JEPA conclusions remain unavailable.|
| Reproducibility |10%|7.5|Counts/budgets/source provenance good; normalization, exact regularizer and calibration can be clearer, compact packet is not a full rerun.|
| Clarity and narrative |10%|8.0|Readable unified question and sequential test logic; some causal shorthand in abstract/headings.|
| Figures |5%|6.0|Useful argument-oriented plots/diagrams; important diagram overlap and misleading physical-mirror contract.|
| Submission fit |5%|7.5|Fixed requested title and anonymized official template; final rendered length, references and compliance remain separate checks.|
| **Weighted total** |**100%**|**73.0 /100**|Revision can fix factual wording and diagrams; evidence limitations must remain bounded.|

No acceptance prediction is implied by these scores. The next version should be rescored on actual corrections, with unchanged research limitations retaining their deductions.
