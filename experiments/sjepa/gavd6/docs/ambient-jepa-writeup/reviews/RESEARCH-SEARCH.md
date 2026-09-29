# Search record and recommendation rationale

28 September 2026. This record supports the appended [research direction](../writeup/RESEARCH_DIRECTION.md). It is a targeted literature investigation, not a systematic review, patent search, or proof of absence. The recommendation is a research hypothesis; no proposed experiment has been run.

## Search scope

The investigation searched arXiv and followed primary sources from PLOS, Journal of Biomechanics, IEEE, CVF, PMLR, Robotics: Science and Systems, author repositories, and the primary manuscripts hosted in PubMed Central. The parent researcher and an independent reviewer investigated overlapping questions separately, exchanged especially close leads, and checked the resulting draft. Sources discovered through secondary indexes were used only after checking a primary source. Coverage includes September 2026 preprints, with preprint status distinguished from publication.

Actual query strings included:

- `"gait" "change" "conformal" markerless`
- `"gait" "paired" "occlusion" measurement uncertainty`
- `"gait" "longitudinal" "change" "markerless" uncertainty`
- `"pose" "preserving" "pathological" reconstruction uncertainty gait`
- `"human pose" "biomechanical" "bounds" uncertainty optimization`
- `"gait" "identifiability" "occlusion"`
- `"monocular" "pose" "partial identification"`
- `"biomechanical" "uncertainty" "feasible" joint angles bounds markerless`
- `"human pose" "ambiguity" "adversarial" uncertainty biomechanics`
- `"gait" "measurement" "counterexample"`
- `"pose estimation" "feasible set" uncertainty human`
- `"gait" "uncertainty" "change" occlusion reconstruction arxiv`
- `"DeepGaitLab" Shin 2026`
- `"SynthGait" "occlusion"`

Queries were followed by title searches and inspection of methods, evaluation, limitations, and appendices in the closest papers. Missing keyword matches were not treated as proof that a method or experiment was absent. In particular, the web text search initially missed SynthGait-19K's occlusion appendix; its PDF and directly downloaded HTML confirmed the experiment.

## Findings that changed the recommendation

The [independent novelty audit](research-direction-novelty-audit.md) provides the detailed overlap analysis. These are the important consequences for scope:

| Initially attractive claim | Finding and resulting decision |
|---|---|
| JEPA features can estimate gait quantities. | SynthGait-19K already includes a V-JEPA2-initialized gait estimator. Treat JEPA as a controlled representation comparison. |
| A model can measure within-person change. | Stenum and DeepGaitLab already evaluate relevant changes against independent references. Require the crossed movement/observation test and a specific failure-detection method. |
| Uncertainty can flag occluded or atypical biomechanics. | Cotton–Sinz and Pace already investigate biomechanical uncertainty and selective retention. Compare against such approaches; preserve the complete support of each measurement. |
| Physics can improve monocular reconstruction. | OpenCap Monocular already combines the relevant stages. Use a contemporary pipeline as a comparator and isolate assumptions that could suppress atypical movement. |
| Task-specific intervals or pose uncertainty sets are new. | Wen et al., CUPS, CLOSURE, and SLUE provide direct methodological precedents. A local trajectory search is an application-specific diagnostic, without certified outer bounds. |

The residual hypothesis is specific: a targeted search for alternative landmark-consistent trajectories may expose misleading per-leg change reports more effectively than simpler uncertainty signals, when tested on observation-only nulls and independently measured real changes under the same visibility conditions. The research must establish that advantage; combining familiar components does not establish novelty or impact by itself.

## Relative priority of the directions in PLAN.md

These judgments concern the next study, not the overall value of each field.

| Direction | Role and reason for priority |
|---|---|
| Preserve per-leg change and detect unsupported comparisons. | Recommended focus. It follows directly from the pooled zero-response result, occlusion sensitivity, and side-assignment failures. A narrowly scoped diagnostic can be tested without new foundation-model training. |
| Calibrated depth and 3D reconstruction. | Immediate enabling work. Known-camera experiments separate the measurement question from unconstrained scale and camera estimation. Extra depth must improve an independent measurement, rather than merely agree with the motion prior. |
| Physical grounding and OpenSim losses. | Begin with explicit kinematic constraints and an offline assumption audit. Joint ranges and continuity can be checked early; forces, assistance, and balance demand a larger reference protocol. Requiring symmetric movement would conflict with the intended measurement. |
| Physics-aware forecasting or robot simulation. | A later branch. v08 restores complete observed windows and gives no evidence about causal prediction. Human assistance and contact introduce assumptions that robot simulations do not validate automatically. |
| Discriminative and generative S-JEPA. | Discriminative measurement remains within the proposed test. Generation would need a separate decoder and verification that requested movement differences survive synthesis. Existing clinical gait generation further weakens a broad novelty claim. |
| Joint generation of people and objects. | Potentially relevant to transfers and balance, but introduces object pose, contact, and support labels. A fixed chair interaction would be a more manageable later target than unconstrained scene generation. |
| Qwen with learned tool selection. | Defer until fixed and rule-based pipelines establish which tools add independent evidence. Later evaluation should trade measurement error and report coverage against latency and participant effort. General tool use or RL is not the present scientific gap. |

## Source checks beyond the appended bibliography

The appended section cites ten especially relevant external sources, in addition to the overview's sources. The independent audit checks further close overlaps. These additional primary sources informed screening; the depth of inspection is stated to avoid implying that every search result received a full-paper review.

| Source | Inspection and relevance |
|---|---|
| [Pemasiri et al., 2026 preprint](https://arxiv.org/html/2603.02499v1), *Biomechanically Accurate Gait Analysis* | Methods and dataset sections inspected. Multiview reconstruction, anatomical markers and OpenSim are established ingredients. |
| [CUPS, ICML 2025](https://proceedings.mlr.press/v267/zhang25g.html) | Publisher abstract inspected by parent; also considered in independent audit. Conformal uncertainty for human pose/shape, including video dependence, is existing methodology. |
| [GAITGen, WACV 2026](https://arxiv.org/abs/2503.22397) | Primary abstract and publication metadata inspected. Pathology-conditioned gait generation already exists. No claim about its absolute biomechanical validity is needed here. |
| [RecovGait, Sensors 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656669/) | Primary indexed text and independent review. Occluded Parkinsonian keypoint restoration is prior work; it does not establish the proposed per-leg 3D change test. |
| [Horsak et al., simulated pathological gait validation, 2025](https://doi.org/10.1016/j.jbiomech.2025.112986) | Primary record inspected by independent reviewer. Instructed pathological patterns in healthy volunteers are relevant controls but distinct from actual clinical gait. |
| [Carvalho et al., older-adult repeatability, 2024](https://run.unl.pt/server/api/core/bitstreams/9d6234fc-0de6-42b5-95de-f80116e63fd0/content) | Primary search record screened. Older-adult markerless repeatability is an existing research topic, not a proposed first contribution. |
| [Ye et al., knee-osteoarthritis repeatability, 2026](https://doi.org/10.1111/os.70431) | Publisher abstract inspected. Clinical repeatability and minimal detectable change also have recent population-specific precedents. |
| [Jariwala, missing-keypoint interpolation, 2026 preprint](https://arxiv.org/abs/2609.09670) | Primary abstract screened. Supports retaining interpolation as a practical control; it is not independent evidence of clinical measurement validity. |

OpenCap Monocular's methods and official [repository](https://github.com/utahmobl/opencap-monocular) were inspected for feasibility. This does not establish that every required model, license, calibration, or identifiable video is available for the proposed study. Its published validation cohort is a feasibility lead, not an untouched confirmation set for its already tuned pipeline.

## Access and inference limits

Some publisher requests timed out or returned a browser check. Closest claims about SynthGait-19K and DeepGaitLab were checked in directly retrieved primary HTML as well as available PDF/indexed primary records. The audit records retrieval hashes and locators. Other papers marked as screened above were not used to support fine-grained absence claims. No search can exclude unpublished work, differently named methods, unindexed papers, or future concurrent work.

The proposed novelty is consequently conditional on implementing the specified test and demonstrating an advantage over strong baselines. It should be refreshed before preregistration or submission. A failure to find a conflicting trajectory cannot justify a reliability certificate, and a lower error on selected easy cases cannot establish usefulness for residents whose movements are harder to observe.

## Decisions made after adversarial review

The [draft review](research-direction-draft-review.md) led to five changes: restrict ambiguity claims to observed landmarks; define change, sub-resolution change, and inconclusive comparisons; separate full-cohort reconstruction tests from selective reporting; make the pilot's independence and change-magnitude limits explicit; and specify compatible 2D encoder inputs for the JEPA comparison. Final dispositions are in [RESEARCH-DISPOSITION.md](RESEARCH-DISPOSITION.md).
