# Version 04: independent world-models and methods review

Reviewed `paper-v04.tex`, its complete diff from v03 and the v04 model diagram. The root reports a compiled nine-page main body; this methods review does not replace its PDF inspection. The separate source-grounded [reproduction contract](../evidence/reproducibility-methods.md) was authored by this reviewer at the writer's request and is not counted as independent evidence of an experiment.

## Findings and disposition

| ID | Severity/status | Evidence | Correction or disposition | Residual limitation |
|---|---|---|---|---|
| W01-4 | Resolved | Main now states median origin, per-axis quantile-span isotropic scale, hidden/reference exclusion, and fixed fallback. | Matches `training.py:288–321`. | Incomplete local raw/checkpoint packet remains disclosed. |
| W01-6 | Substantially resolved | Main now gives `.05×VICReg`, its 25/25/1 weights, scalar/dense calibration equation, AdamW settings, warmup/cosine and clipping. The reproduction contract specifies exact reductions, translations, teacher/center and coefficient rules. | Keep supporting file linked in README and audit ledger. Optional high-value main addition: the shared feature-response coefficient, distinct from repair calibration. | No convergence or coefficient sweep is available. |
| W01-5 | Resolved | The v04 diagram's shortened loss label fits; the lower path now says single-state encoder/readout and caption explains paired training compares two outputs. | Diagram accurately separates training targets, EMA updates and deployment. | A verified observed restoration example remains unavailable in the transferred packet. |
| W03-1 | Bounded supporting detail | Endpoint-dependent masked normalization is now explicit in the reproduction contract. Main does not equate latent residual units with physical motion. | Existing support-level explanation is sufficient at nine-page scope. | Latent response still includes context/normalization effects and has no physical-unit guarantee. |
| W04-1 | Minor reproducibility, optional | Main has shared training-calibrated feature weight but only prints the repair formula. Source response calibration is `lambdaJ=.1 sqrt(Gbase/max(Gdelta,GE))`, common to both auxiliaries/all seeds, based on 32 initial training batches (`response_calibration.py:219–230`). | Add this compact formula if space permits; otherwise the linked supporting record is exact. Do not say each auxiliary separately has 10% of base gradient. | Matching is initial gradient magnitude only. |
| W04-2 | Minor language | “projected translated encoder means” can mean either the learned projector head or camera projection, both used nearby. | Prefer “a projector applied to translated-view mean encoder features” if rewriting. | Meaning is source-correct; no factual correction required. |

The probe population is now explicit: 333 diagnostic pairs from 112 people already seen during encoder pretraining. This removes a possible interpretation of probe folds as independent encoder generalization. Reaggregation weights 4:1 for endpoint metrics and 2:1 for nonzero responses are also documented without changing the inferential unit.

All previous factual corrections are retained. The numerical degree parameters, paired scalar/dense definitions, support, mask/inference semantics, and representation/dynamics boundaries remain consistent with code. No unresolved material method misstatement was found in v04. The closest-work paragraph accurately situates the adaptation rather than claiming a new general JEPA mechanism.

## Fixed rubric

| Dimension | Weight | Score /10 | Concrete reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution |20%|6.5|Same bounded empirical contribution and no independent confirmation.|
| Claim accuracy and evidence support |20%|9.0|Exact source-consistent methods; unsupported broader conclusions remain excluded.|
| Evaluation and statistical rigor |15%|7.0|No new independent evidence; adaptive 14-person cohort and three-seed limitation persist.|
| Scientific insight and positioning |15%|8.0|Good controls and close-work scope; no unique causal or generic representation mechanism.|
| Reproducibility |10%|8.5|Main and supporting contract now explain critical numerical operations; full source rerun still needs remote/licensed assets and checkpoints.|
| Clarity and narrative |10%|8.5|Pairwise fitting versus single-state deployment clarified; technical load remains manageable.|
| Figures |5%|8.5|Architecture semantics and label fit improved; final-size QA and absent real reconstruction remain limits.|
| Submission fit |5%|8.0|Compiled main length reported as nine pages; complete submission package/anonymity checks remain integration tasks.|
| **Weighted total** |**100%**|**78.75 /100**|Reproducibility and diagram corrections earn the increase; unchanged empirical scope does not.|

The remaining optional edits can improve precision. They should not be presented as solutions to absent external baselines, convergence analysis, independent confirmation or real-world validation.
