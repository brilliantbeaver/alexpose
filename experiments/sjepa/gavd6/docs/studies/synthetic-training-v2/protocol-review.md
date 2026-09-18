# Independent Codex protocol review

**Reviewer:** separate Codex literature-challenger agent context; not the coordinator or protocol author. **Date:** 18 September 2026. **Reviewed artifact:** [protocol.md](protocol.md), 45-line initial snapshot, SHA-256 `8b29c037072470e3135577d81a5f6dfa52b6a6d8ad1205c22dfab24416be6ec0`. References below identify line numbers in that snapshot, which may change after correction. Also inspected the implementation request and [primary literature](literature.md). This review did not inspect new model code, fitted checkpoints or empirical predictions and does not substitute for implementation/results review.

**Verdict:** suitable to begin explicitly nonempirical CPU fixture validation after this review. The high-priority issues below must be resolved in a versioned protocol/configuration before an empirical screen can yield `pass` or `fail`; otherwise return `insufficient_evidence`. No scientific result or novelty claim is currently established. These are specification findings, not demonstrated code defects.

## Prioritized findings

### R1 — High: Gate B is not yet an executable decision rule

**Evidence:** snapshot lines 31–35 name several motion endpoints, provisional 2% coordinate/5% displacement/<1% clean thresholds, adequate support and multiple seeds, but do not define the reference comparator for each margin, how uncertainty determines a verdict, what constitutes adequate support, or preservation margins for amplitude and timing. The companion plan requires a predeclared accuracy–preservation tradeoff and explicit `pass`, `fail` or `insufficient_evidence` outputs.

**Consequence:** two analysts could select different comparators/endpoints or classify the same result differently. Mean displacement improvement could coexist with unacceptable asymmetry or timing attenuation. Leaving confirmation margins to development is appropriate, but does not define a development screen by itself.

**Smallest correction:** name the decisive contrast and selected practical comparator; write a conjunction of development conditions with direction, reference denominator and uncertainty rule. Specify minimum supported independent groups and seeds, and how missing support leads to `insufficient_evidence`. If amplitude/timing margins cannot yet be justified, prohibit a preservation `pass` until an explicit development calibration establishes them. Freeze each revision before its next screen; do not derive rules retrospectively from a favorable candidate.

### R2 — High: Motion measurements need exact definitions and nondegenerate denominators

**Evidence:** snapshot line 31 gives signed ankle separation, 0.20-second displacement, amplitude and event timing without formulas, anatomical-axis convention, reference normalization, valid support or event definition. Lines 9/11 separately distinguish input normalization and reference-box evaluation normalization. Their relationship for motion scoring is not stated.

**Consequence:** independently rescaling output/reference trajectories can conceal amplitude attenuation; a signed group average can cancel left/right errors; near-zero clean error makes relative degradation unstable. Event estimates from partial tracks can look precise without valid reference events. [PoseBERT's analysis](https://arxiv.org/html/2208.10211v2) explicitly reports deterioration of already-good predictions, so aggregate error alone is insufficient protection.

**Smallest correction:** freeze the coordinate frame and common reference denominator; invert the single noisy-input-derived normalization before evaluation, never fit a new one to candidate outputs or clean targets. Define separation error without cancellation, displacement at actual 0.20-second elapsed time, amplitude estimator and event-matching rule without time warping. Name low-amplitude/near-zero-error handling and minimum valid trajectory/event support. Mark unavailable amplitude/events as pending rather than treating missing values as preservation success.

### R3 — High: Target/query/loss support is underspecified for naturally missing inputs

**Evidence:** snapshot lines 9 and 19 correctly permit missing-input/valid-target queries and pair clean/noisy target arms. Line 23 masks 50% of eligible input tokens but does not explicitly state which fixed output queries receive clean supervision, whether naturally absent observations are included, or whether clean teacher and coordinate targets use exactly the same noisy-input-derived transform. The candidate's stop-gradient boundary is not stated explicitly.

**Consequence:** an implementation could train only artificially hidden observed joints yet advertise missing-joint restoration, leak clean support through teacher-side inference wiring, or compare latent and coordinate arms with different supervised support. [S-JEPA's methods](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) mask target embeddings after full-sequence target encoding; its action-recognition formulation does not automatically resolve missing-input restoration.

**Smallest correction:** define separate observed, artificial-hidden and target-valid masks; a fixed time×joint query grid; precise loss-support intersections for each arm; and common clean-target support for the objective contrast. Specify clean teacher input, stop-gradient and shared EMA/centering behavior. Give coordinate and latent targets the same transform derived from noisy observed inputs. Require a missing-input/valid-target fixture and target-mask mutation test showing identical inference inputs/predictions. These fixtures may proceed before empirical screening.

### R4 — Medium: The practical MLP's training objective and tradeoff allowance are not frozen

**Evidence:** snapshot line 19 names a SmoothNet-style MLP and filter strengths, while lines 23–25 principally specify transformer settings and two learning rates. MLP topology, loss, masking/support adapter and smoothing/regularization settings are not identified. There is no learned-method preservation tradeoff sweep.

**Consequence:** a nominally strong comparator could become a weak arbitrary MLP, while the JEPA arm receives more effective tuning. [SmoothNet](https://arxiv.org/pdf/2112.13715) uses reference-position and reference-acceleration losses, with a specifically temporal architecture. [SynSP](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_SynSP_Synergy_of_Smoothness_and_Precision_in_Pose_Sequences_Refinement_CVPR_2024_paper.html) makes the precision/smoothness tradeoff itself central.

**Smallest correction:** record the exact body-12 MLP architecture, missingness/confidence/time adapter, objective and a small development-only regularization grid. Count its tuning budget. Compare the same window, support and latency and save the accuracy–preservation curve. A simpler deliberate adaptation is acceptable when labeled; an official-reproduction claim is not. Adding PS-Mamba/diffusion dependencies is not required for the first milestone.

### R5 — Medium: Generalization and unusual-motion strata remain declarations, not frozen assignments

**Evidence:** snapshot lines 13/15 propose people counts, six conditions and an excluded extractor family without naming assignments or exposure records. Line 29 names clean/accurate strata but not unusual-motion or side-asymmetric strata, despite the question at line 3. No real-data preservation claim is presently made, correctly.

**Consequence:** later choosing the excluded family or nuisance combinations based on observed performance would invalidate a held-out claim. Clean tracks alone may not reveal suppression of unusual but valid movement.

**Smallest correction:** before source fitting, save named source extractors, excluded family, motion/person groups and nuisance holdout combinations with exposure status and content hashes. Declare unusual-motion stress strata or explicitly keep that preservation claim pending. Synthetic asymmetry/time perturbations are sensitivity checks, not diagnoses. Unknown prior exposure must continue to prevent confirmation.

### R6 — Low: Dense parameter prose and joined words reduce reviewability

**Evidence:** examples include `strengths0,1,2`, `width96`, `AdamW3e-4`, `Mask50%`, `seeds17/29/43`, `The48` and `at least20%` in snapshot lines 19–43. The basic question and claim boundary are readable, but the joined tokens make settings and units harder to audit.

**Smallest correction:** insert normal spacing and put the handful of training/margin parameters in a compact table or configuration link. Define visible as reference visibility rather than estimator confidence. Retain the explicit distinction between supervised synthetic clean targets, independent real references and fixture evidence. Avoid introducing world-model or clinical language.

## Strongest competing explanation

Any future paired-JEPA gain may be explained by access to cleaner targets, additional training exposure, smoothing or readout capacity rather than latent prediction. The protocol's same-clean-information coordinate representation arm directly addresses the largest confound. Its practical MLP, direct denoiser, initialized encoder and unchanged-input baselines address the rest only once actual masks, loss support, tuning and costs are matched and recorded. Low latent loss, high effective rank or smooth output cannot establish the intended measurement result.

## Claim verdicts

| Proposed claim | Current verdict | Evidence required |
| --- | --- | --- |
| Software implements the requested contracts | Not reviewed here | Inspect implementation and adversarial contract fixtures. |
| Synthetic pairs improve proxy pose accuracy | Insufficient evidence | Saved matched predictions, person-balanced errors and independent support counts. |
| JEPA improves beyond equally informed coordinate learning | Insufficient evidence | R1–R4 resolved; matched clean-target contrast, multiple seeds and declared uncertainty. |
| Restoration preserves timing/anatomical sides | Insufficient evidence | R1/R2/R5 resolved; reference-based motion metrics and noninferiority rules. |
| Improvement transfers across extractors/nuisances | Insufficient evidence | Frozen exclusions/exposure records plus per-held-extractor results. |
| Real anatomical/temporal preservation | Pending; cannot follow from synthetic/fixture results | Independent real annotations, blinded assessment and protected confirmation. |
| First temporal refiner / masked-MoCap system / gait JEPA | Reject broad novelty wording | Closest papers already overlap; see literature ledger. |
| New narrow empirical contribution | Hypothesis | Reproducible restoration–preservation evidence beyond matched baselines; literature search alone cannot establish it. |
| Personalized teaching is useful | Not supported by the old panel | Fresh development headroom, followed by cost-matched scene/state/source-progress comparisons. |

## Controls already specified correctly

The protocol explicitly limits the task to offline same-camera 2D restoration, calls SMPL-H landmarks a synthetic proxy, keeps privileged metadata outside inference, identifies privileged boxes, groups motion variants and people, separates development from confirmation, budgets pretraining plus readout, and preserves independent branch STOPs. These are useful controls, not findings of empirical success.

The coordinator owns the subsequent `review-dispositions.md` record. This review remains a snapshot. Material corrections require a recorded follow-up verification; it must not be silently rewritten to imply that the original protocol had no issues.

## Follow-up inspection of protocol clarifications

On 18 September 2026 the same reviewer read the actual 66-line revised [protocol](protocol.md), SHA-256 `fb9e9bac6cfd4e9236430df41d7124b5acddd32663aad8b496b6f843dcb56a33`, including its **Pre-fit implementation clarifications following independent review**. The original findings above remain intact. This follow-up checks the specification text, not whether source executions implement it.

| Finding | Follow-up status | Direct evidence and remaining requirement |
|---|---|---|
| R1: executable decision rule | Specification structure corrected; calibration and empirical adjudication pending | The added source-adjudication section names all four decisive comparators, a conjunction across declared extractor/seed strata, relative primary effect and paired-interval rule, displacement, clean retention, supported amplitude/timing bounds, minimum groups/seeds and missing-support outcomes. It requires a pre-fit decision specification tied to a calibration artifact and run identity. No completed calibration or numeric amplitude/timing margins were supplied. At least two groups is expressly a software minimum, not adequate statistical power. Without that specification/support, return `insufficient_evidence`; this correction does not justify a source or real-preservation pass. |
| R2: motion definitions | Textual specification corrected | The revision fixes shared noisy-input geometry, inversion before pixel scoring, prediction-independent reference-box normalization, exact elapsed-time displacement, absolute signed-separation error, demeaned RMS amplitude, zero-amplitude/clean denominators, full bilateral support and ordered peak matching. It explicitly calls the peaks operational 2D ankle-separation maxima rather than heel strikes and leaves cadence unestimated. The implementation/results review must verify those formulas against saved arrays; real measurement validity still needs independent temporal references. |
| R3: target/query/loss support | Textual specification corrected | Artificially hidden observed tokens and naturally missing-input queries are separately named, with their union intersected with valid clean support for the coordinate/paired contrast. Fixed output queries do not use target masks. The teacher uses the shared noisy-input transform with detached embeddings. Ordinary-JEPA support is disclosed as a supervision-source difference. The local centered-CE plus VICReg recipe is named rather than passed off as official S-JEPA. Contract tests remain the implementation evidence. |
| R4: practical baseline | Design allowance corrected; empirical tradeoff pending | The practical section specifies shared per-joint temporal MLP dimensions, auxiliary inputs, residual/absolute adapter, L1 reference-position plus reference-acceleration loss, acceleration weights 0/0.1 and matching learning-rate allowance. Actual tuning cost and the tradeoff curve still require execution. It is correctly labeled a SmoothNet-style adaptation, not reproduction. |
| R5: held conditions and unusual motion | Partly specified; actual source assignments pending | The initial adapter now names clean, blur, obstruction and held development blur+obstruction, fixed camera, and a smaller scope than the six-condition proposal. Actual eligible people/motions, extractor-family exclusions, exposure identities and authoritative manifest hashes still must be produced before source fitting. Unusual-motion/anatomical strata are explicitly pending. No fixture can establish that a chosen family was previously unseen. |
| R6: readability | Improved with minor spacing debt | Most dense joined settings now have spaces. A few examples remain, including `patch4`, `at most2,000`, `by1` and `limbs,95th`. These do not change the decision rule, but can be cleaned before circulation. Detailed formulas now clarify visibility and unit interpretation. |

**Updated verdict:** the revised text resolves the earlier missing formulas, query boundary and practical-baseline design details, and supplies the structure of an executable development decision. It does **not** supply independently calibrated margins, named source assignments, sufficient sample size, completed source comparisons or real references. CPU fixtures may validate those contracts; scientific effect, motion preservation, transfer and narrow novelty remain unestablished. The strongest competing explanation and original claim verdicts are unchanged.
