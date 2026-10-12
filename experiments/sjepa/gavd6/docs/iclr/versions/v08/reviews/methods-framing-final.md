# v08 independent methods and scientific-framing review

Reviewed 2026-09-26, following `methods-framing-audit.md`. Scope: the revised main source and appendix, the completed method implementation, retained validation record, and new participant-profile exports. This is an adversarial manuscript audit, not a new training run, notebook execution, or independent empirical validation. The reviewer did not edit the manuscript or scientific code.

## Verdict

The revision now gives a defensible motivation → operational comparisons → development findings narrative. Self-supervised learning and JEPA motivate the method, while the text expressly identifies privileged projected supervision, a complete-window restoration task, and the absence of future-state prediction. The retrospective nulls, adaptive reuse of development people, and notebook execution limits are stated accurately. No additional experiment is needed to make those boundaries honest.

The initial review identified three concrete terminology corrections concerning fixed array slots versus true anatomical identity, observation availability versus reference visibility, and person separation versus independent confirmation. All three have now been corrected and independently checked in the source. No unresolved material methodological misstatement remains in the new framing. The study remains a small, adaptive development evaluation rather than evidence of a validated gait world model or clinical restoration system.

Resolution check on 2026-09-26, restricted to the three edited passages; no training, notebook execution, or PDF-layout recheck was performed:

| Source | SHA-256 at resolution check |
|---|---|
| `paper-v08.tex` | `2f0b53bf31344eb52f59ecb3878beecf0d6a76f52de8b11c43505238230852d2` |
| `appendix.tex` | `cec97469bf015803bc0077be84d62af7c49cbfba1488228b8f7204c9ca86009c` |

## Actionable findings

### MF1 — Fixed joint slots are not preserved observed anatomical identity

**Severity: moderate; resolved.** The initially reviewed `paper-v08.tex:38` said each state retains the same coordinate array and “joint identities” through corruption. The naming intervention intentionally exchanges which anatomical coordinate occupies each named input slot. The array *shape*, slot ordering, and timestamp ordering are preserved; the correctness of input anatomical naming is not. The corrected line now explicitly says “coordinate-array shape, joint-slot order, and timestamp order”; the later naming paragraph explains the altered input correspondence and unchanged reference names. This resolves the objection.

**Evidence:** `src/gavd6_sjepa/research_directions/gait_fidelity/data.py:302–311` copies each input and permutes `xy`, `confidence`, and `observed` by `SWAP`, preserving timestamps. `synthetic_training_v2/models.py:116–125,138–141` preserves the joint/time indexing through tokenization and decoding. The manuscript correctly explains unchanged reference names later, so this is an internal wording inconsistency rather than a flaw in the study.

**Concrete correction:** “Each state retains the same 128×12×2 coordinate-array shape, joint-slot order, and timestamp order through corruption and restoration. Naming corruption can place the opposite anatomy's coordinates in those fixed slots; anatomically named references remain unchanged.” The shorter first sentence plus the existing later naming explanation is also sufficient.

**Residual limit:** Fixed indexing does not itself establish anatomical recovery; the separate assignment diagnostic remains necessary.

### MF2 — Normalization is based on observation availability, not reference visibility

**Severity: moderate; resolved.** The initially reviewed `paper-v08.tex:53` called the normalization inputs “visible, unmasked observations,” and the masking/scaling row at `appendix.tex:260` repeated “visible.” Here `visible` is already a distinct reference property; a pose estimator can emit an available estimate for an image-occluded joint. Using “visible” risked implying that ground-truth image visibility enters the input normalization. Both passages now say “available, artificially unmasked observations,” and the appendix now explicitly calls its per-window operation “Inference normalization.” This resolves the objection without conflating input scaling with evaluation NLE.

**Evidence:** `gait_fidelity/training.py:288–321` uses only `inputs['observed']`, intersected with the inverse artificial mask. No target visibility is read. `gait_fidelity/data.py:99–120` keeps input `observed` and target validation separate. Appendix A's own formula already correctly defines context as `C_i = O_i ∧ ¬M_i`.

**Concrete correction:** Use “observed and artificially unmasked coordinates,” or “available, unmasked observations,” consistently. Change “Evaluation normalization uses its own observed window” in the same appendix row to “At inference, input normalization uses that window's observations.” This avoids conflating input normalization with evaluation NLE's reference-box denominator, which the manuscript correctly specifies elsewhere.

**Residual limit:** This supports the narrow claim that hidden coordinate values and target geometry do not enter the input transform statistics. It does not imply that reference-supervised training is self-supervised or that development was never inspected.

### MF3 — Held-condition evaluation is not an additional confirmation cohort

**Severity: minor but scientifically useful; resolved.** The initially reviewed `appendix.tex:269` said the held 15° and ViTPose results are “not independent-person generalization.” The development people are correctly separated from fitting people; the intended limitation is that holding conditions out does not create an additional untouched person-level test. The corrected paragraph identifies 14 different development people and states that these held conditions do not provide an additional independent confirmation cohort. This resolves the objection.

**Evidence:** `gait_fidelity/cohort.py:128–152` requires one original split per approved identity and rejects identical motion content crossing identities or splits. Completed evaluation contains 14 people different from the 112 fitting people, but all follow-ups reuse those same development people. The current main text accurately describes this.

**Concrete correction:** “The 15° and ViTPose results test condition shifts within the reused development cohort, rather than providing an additional independent confirmation cohort.”

**Residual limit:** Known-identity separation does not prove absence of unknown aliases, upstream estimator pretraining overlap, or adaptive development selection. Those gaps are already disclosed appropriately.

## Accepted scientific boundaries

- **SSL, JEPA, and world models:** The text explains the connection rather than claiming that the completed system has learned future dynamics. Both the abstract and introduction disclose paired synthetic supervision; the introduction explicitly identifies the privileged reference teacher and supervised coordinate readout. The bidirectional full-window implementation supports the offline-restoration description. “Measurement requirements for future gait world models” is a defensible motivation/inference, not an experimentally established capability. Keep the explicit future-state limitation in the abstract and discussion.
- **Nulls and provenance:** `G = error_comparator − error_candidate`, positive in favor of the candidate, is consistent throughout. `H0: E[G] ≤ 0` and the stage-specific alternatives are clearly marked as retrospective formalization. The manuscript neither claims that these equations were originally preregistered nor converts the retained two-sided intervals into newly conducted one-sided tests. It does not accept the null, claim equivalence, or invent a preservation/noninferiority margin.
- **Comparison sequence:** Core JEPA/change versus direct/change is the initial recorded primary; delta/change versus endpoint/change is the next primary; delta/dense versus delta/low-scalar on ViTPose waveform error is the repair primary. The stronger direct/coordinate arm is not substituted into the original primary. The same 14 people and three fitted seeds remain explicit, and the source stages are not presented as independent confirmations.
- **Inferential scope:** Core/response crossed-person-and-seed intervals and repair's seed-conditional person-t interval remain distinct. Newly computed profiles, assignment effects, stratification, penalty sensitivity, and other intervals are exploratory and unadjusted. The discussion explicitly states that neither resampling approach accounts for adaptive selection of later studies.
- **Training and data boundaries:** The revision separates raw-motion preprocessing from model-interface shape preservation. Resampling, window selection, synthetic editing, mirroring, camera projection, and corruption alter values or row counts; they do not imply untouched raw data. Fixed `[B,128,12,2]` input/output shapes and the `[B,384,96]` token representation match the code. The four-frame regrouping and inverse decoder do not truncate, interpolate, or time-warp either leg.
- **Leakage claims:** Student input allow-lists, training-only pair selection and coefficient calibration, exclusion of the held estimator/edit from fitting, source-person split inheritance, and locked-confirmation rejection are stated concretely. The caveat that ordinary bundle validation may read development references is correct and avoids the false stronger assertion that development targets are never opened. Reference targets do not supply optimization/calibration losses. Known-identity and upstream-pretraining limits are explicit.
- **Masking:** All 384 slots remain; artificial hiding zeros coordinates, confidence, and usable-observation channels. Natural absence may retain finite native confidence, as the new integrity appendix correctly notes. No target-derived padding mask removes missing queries. The implementation-derived normalization claim is now accurate after MF2's terminology correction.
- **Completed notebooks versus experimental evidence:** The new map correctly distinguishes source runs, instructional CPU calculations, retained-result reconstruction, and planned controls. Only graph-time source fits are completed among the mask comparisons. The validation receipt (`docs/studies/gait-fidelity/records/tutorial-refresh-20260925.json`) explicitly records `TARGETED_VALIDATION_PASSED_FULL_FIXTURE_BLOCKED` and `haic_validation: false`; the appendix does not claim all current notebooks were freshly executed. “Remote accelerator validation” here refers to notebook validation, not denial of the retained source training receipts; adding “notebook” would make that sentence still clearer but is not a material objection.
- **Optimization and causal restraint:** The exact shared-coefficient calibration, initial endpoint 10% versus delta 0.0013% gradient ratio, and near-universal clipping remain qualified as initialization/update-ledger facts. They are not presented as the persistent influence of a term, an AdamW update decomposition, or a causal explanation for final scores. The compound original change objective remains distinct from the low-scalar-versus-dense repair.
- **Participant profiles:** The new CSV contains all 14 people in every panel. Direct improves pooled response over unchanged inputs for 10/14 people, beats zero response for 14/14 in clear conditions and 1/14 in occlusion. Seed ranges are labelled as fitted variation, not confidence intervals. The figures do not imply recovered clinical status or reconstructed source-pose examples. The abstract's zero-minus-direct result, 1.733325° [0.528417, 3.149885]°, matches the retained audit, including interval sign reversal.

## Expository assessment

The new introduction gives a natural reason to distinguish predictable features, decoded trajectories, and anatomical names. The stage sequence is easier to follow because each follow-up answers a narrower question exposed by the preceding result. The appendix carries software and data-contract detail without turning the main narrative into a notebook execution log. The clinical examples remain motivation and are not used to label the synthetic edits as disease or treatment effects.

The main contribution remains an auditable diagnostic evaluation of these fitted procedures. Additional world-model vocabulary does not strengthen the empirical evidence or make the general insight that metrics require fidelity novel. The paper's substantive value is the combination of paired supervision controls, uncertainty, score/failure decomposition, supervision-weight sensitivity, and a separately evaluated anatomical assignment discrepancy. The revised text generally presents that combination faithfully.

No request is made for new training, new hypothesis tests, additional subjects, clinical experiments, or fabricated pose examples. Those are potential future studies, not editorial conditions for accurate reporting of the completed work.
