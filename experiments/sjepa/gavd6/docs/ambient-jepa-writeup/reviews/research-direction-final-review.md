# Final independent review of the research-direction appendix

28 September 2026. Reviewed `writeup/RESEARCH_DIRECTION.md`, SHA-256 `6c1dfd0530a8eec5d61c10e619027c82ed3d6fbf8543af115473c3365e6061d4`.

**Verdict: no remaining substantive blocker to delivering this research proposal.** All five required revisions from the draft review are resolved. This verdict concerns the proposal's scientific framing and coherence; it does not establish that the method works or has cleared an exhaustive novelty search.

| Previous issue | Resolution checked in the revised draft |
|---|---|
| Observation evidence overstated | Lines 15 and 39 restrict alternatives to visible 2D landmark detections. Hidden estimates remain unobserved, and the text acknowledges that silhouettes could reject an alternative. |
| Unsupported versus unchanged decision undefined | Line 47 now distinguishes intervals beyond the resolution band, intervals wholly inside it, and inconclusive cases. Calibration is joint across legs and accounts for repeated pairs. Whole-pair reporting preserves measurement support. |
| Comparative testing hierarchy ambiguous | Line 49 separates full reference-eligible-cohort reconstruction tests from selective reporting. Change-error noninferiority precedes trajectory superiority; failures remain accounted for, zero/interpolation controls remain, and uncertainty resamples people with their pairs. Selective methods are compared at matched retention with difficult-case audits. |
| Pilot independence and target variation insufficiently specified | Line 33 identifies the ten-person healthy pilot, potentially insufficient knee variation, prior tuning of the comparator, and the need for independent final data. Pair composition and change magnitudes must be prespecified. |
| JEPA transfer interface unclear | Line 45 now specifies a shared compatible 2D encoder and 3D readout with fixed 3D initialization, varying JEPA pretraining before identical supervised adaptation. Pretraining resources are reported separately. |

The main claims remain appropriately conditional. The method is an offline search for admissible counterexamples; unsuccessful search is not presented as proof of identifiability. Kinematic constraints are distinguished from balance or force validation. Close prior work substantially narrows the originality claim, and the text does not imply demonstrated benefit for Sequoias residents, fall prediction or clinical decisions. Later Qwen/RL and generative work remains subordinate to a validated measurement need.

The study still needs a preregistered numerical margin, retention policy, failure costs, calibration procedure, sample size and dataset access verification before execution. These are explicit protocol dependencies rather than defects requiring expansion of this four-page proposal.

**Audit scope:** I reread the complete revised appendix and checked `reviews/RESEARCH-SEARCH.md`, including its treatment of prior work and retrieval limitations. Primary-source checks from the earlier reviews remain applicable; no new broad search was performed. I did not run the proposed methods or independently inspect the final figure, PDF pagination, original-overview preservation, or rendered link behavior. The parent researcher is handling those production checks. No manuscript or asset was edited during this closeout.
